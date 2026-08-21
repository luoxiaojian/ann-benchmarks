"""Focused ann-benchmarks adapters for the zvec Vamana Pareto grids.

The active benchmark configuration uses ``ZvecAnnBenchDocIds``. The direct
``ZvecFastQueryDocIds`` and public ``ZvecQuery`` adapters are retained so the
reviewed SIFT configuration can still be selected. All adapters share the same
Vamana build and query parsing. Legacy HNSW, fast-query, coupled-argument, and
old-wheel compatibility code is out of scope.
"""

from __future__ import annotations

import gc
import os
import shutil

import numpy as np
import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    Doc,
    LogLevel,
    OptimizeOption,
    Query,
    VamanaQueryParam,
    VectorSchema,
    create_and_open,
    open as zvec_open,
)
from zvec.model.param import VamanaIndexParam
from zvec.typing import DataType, MetricType, QuantizeType

from ..base.module import BaseANN


# ann-benchmarks pins each container to one CPU. Keep zvec's worker setup
# aligned with that cpuset and avoid binding query threads from inside zvec.
try:
    zvec.init(
        log_level=LogLevel.WARN,
        query_thread_binding=False,
        optimize_threads=1,
        optimize_thread_binding=False,
    )
except RuntimeError:
    # zvec can already be initialized during local validation.
    pass


VECTOR_FIELD = "vector"

_QUANTIZE = {
    "int8": QuantizeType.INT8,
    "uniform_int8": QuantizeType.UNIFORM_INT8,
    "uniform_uint8": QuantizeType.UNIFORM_UINT8,
    "uniform_uint4": QuantizeType.UNIFORM_UINT4,
}

_FLAT_DATA_TYPE = {
    "inherit": DataType.UNDEFINED,
    "uint8": DataType.VECTOR_UINT8,
    "fp16": DataType.VECTOR_FP16,
}

_METHOD_PARAM_KEYS = {
    "index",
    "max_degree",
    "search_list_size",
    "alpha",
    "two_pass_build",
    "reverse_prune_batch_size",
    "use_bulk_build",
    "quantize",
    "use_flat_contiguous_memory",
    "flat_data_type",
}


class ZvecBase(BaseANN):
    """Shared Vamana construction and ann-benchmarks query plumbing."""

    interface = "base"

    def __init__(self, metric: str, dim: int, method_param: dict):
        method_param = dict(method_param)
        unknown = sorted(set(method_param) - _METHOD_PARAM_KEYS)
        if unknown:
            raise ValueError(f"[zvec] unsupported method parameters: {unknown}")
        if metric != "euclidean":
            raise ValueError("[zvec] this benchmark adapter supports euclidean only")
        if method_param.get("index") != "vamana":
            raise ValueError("[zvec] this benchmark adapter supports Vamana only")

        self._metric = MetricType.L2
        self._dim = int(dim)
        self._method_param = method_param
        self._max_degree = int(method_param["max_degree"])
        self._search_list_size = int(method_param["search_list_size"])
        self._alpha = float(method_param["alpha"])
        self._two_pass_build = method_param.get("two_pass_build")
        self._reverse_prune_batch_size = int(method_param.get("reverse_prune_batch_size", 1))
        self._use_bulk_build = bool(method_param.get("use_bulk_build", True))

        self._quantize_name = str(method_param.get("quantize", "int8")).lower()
        if self._quantize_name not in _QUANTIZE:
            raise ValueError(f"[zvec] unsupported quantize value: {self._quantize_name}")
        self._quantize = _QUANTIZE[self._quantize_name]

        self._use_flat_contiguous_memory = bool(method_param.get("use_flat_contiguous_memory", False))
        self._flat_data_type_name = str(method_param.get("flat_data_type", "inherit")).lower()
        if self._flat_data_type_name not in _FLAT_DATA_TYPE:
            raise ValueError(f"[zvec] unsupported flat_data_type: {self._flat_data_type_name}")
        self._flat_data_type = _FLAT_DATA_TYPE[self._flat_data_type_name]

        if self._dim <= 0 or self._max_degree <= 0 or self._search_list_size <= 0:
            raise ValueError("[zvec] dimension and Vamana sizes must be positive")
        if self._alpha <= 0 or self._reverse_prune_batch_size <= 0:
            raise ValueError("[zvec] alpha and reverse_prune_batch_size must be positive")
        if self._flat_data_type_name != "inherit" and not self._use_flat_contiguous_memory:
            raise ValueError("[zvec] native Flat storage requires use_flat_contiguous_memory=true")

        self._using_refiner = False
        self._candidate_topk = None
        self._prefetch = {}
        self._ef = self._search_list_size
        self._query_param = self._make_query_param()
        self._label = self.interface
        self._collection = None
        self._raw_obj = None
        self._path = os.path.join(
            "zvec_indices",
            f"{self.interface}_vamana_euclidean_d{self._dim}_"
            f"R{self._max_degree}_L{self._search_list_size}_"
            f"B{self._reverse_prune_batch_size}_bulk{int(self._use_bulk_build)}_"
            f"a{self._alpha}_cm1_fcm{int(self._use_flat_contiguous_memory)}_"
            f"fdt{self._flat_data_type_name}_{self._quantize_name}",
        )
        self.name = f"zvec-{self.interface}({method_param})"

        self._q = None
        self._n = 0
        self._res = None

    def _make_index_param(self) -> VamanaIndexParam:
        kwargs = dict(
            metric_type=self._metric,
            max_degree=self._max_degree,
            search_list_size=self._search_list_size,
            alpha=self._alpha,
            reverse_prune_batch_size=self._reverse_prune_batch_size,
            use_bulk_build=self._use_bulk_build,
            use_contiguous_memory=True,
            quantize_type=self._quantize,
            use_flat_contiguous_memory=self._use_flat_contiguous_memory,
            flat_data_type=self._flat_data_type,
        )
        # Omission deliberately selects the wheel's reviewed default (two-pass
        # in dev39); config-sift.yml retains its explicit historical setting.
        if self._two_pass_build is not None:
            kwargs["two_pass_build"] = bool(self._two_pass_build)
        return VamanaIndexParam(**kwargs)

    def _make_query_param(self) -> VamanaQueryParam:
        kwargs = dict(
            ef_search=self._ef,
            is_using_refiner=self._using_refiner,
        )
        # Current config.yml omits this field and therefore keeps dev39's
        # schema-aware auto-prefetch. The reviewed SIFT grid passes explicit
        # values through extra_params.
        if self._prefetch:
            kwargs["extra_params"] = dict(self._prefetch)
        return VamanaQueryParam(**kwargs)

    def _refresh_name(self) -> None:
        details = (
            f"{self._method_param}, ef={self._ef}, "
            f"candidate_topk={self._candidate_topk}, prefetch={self._prefetch}"
        )
        self.name = f"zvec-{self._label}({details})"

    def _open_readonly(self) -> None:
        self._collection = zvec_open(
            self._path,
            CollectionOption(read_only=True, enable_mmap=True),
        )
        self._raw_obj = self._collection._obj

    def fit(self, X: np.ndarray) -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        if os.path.exists(self._path):
            shutil.rmtree(self._path)
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

        schema = CollectionSchema(
            name=f"annb_{self.interface}",
            fields=[],
            vectors=[
                VectorSchema(
                    VECTOR_FIELD,
                    DataType.VECTOR_FP32,
                    dimension=self._dim,
                    index_param=self._make_index_param(),
                )
            ],
        )
        build_col = create_and_open(
            path=self._path,
            schema=schema,
            option=CollectionOption(read_only=False, enable_mmap=True),
        )

        for start in range(0, len(X), 1024):
            end = min(start + 1024, len(X))
            docs = [Doc(id=str(i), vectors={VECTOR_FIELD: X[i].tolist()}) for i in range(start, end)]
            results = build_col.insert(docs)
            if not isinstance(results, list):
                results = [results]
            for result in results:
                if not result.ok():
                    raise RuntimeError(f"[zvec] insert failed: {result.code()}")

        build_col.optimize(option=OptimizeOption())
        build_col = None
        gc.collect()
        self._open_readonly()

    @staticmethod
    def _validate_prefetch(prefetch: dict) -> dict:
        if set(prefetch) != {"prefetch_offset", "prefetch_lines"}:
            raise ValueError("[zvec] prefetch must contain exactly prefetch_offset and prefetch_lines")
        values = {key: int(value) for key, value in prefetch.items()}
        if any(value < 0 or value > 256 for value in values.values()):
            raise ValueError("[zvec] prefetch values must be in [0, 256]")
        return values

    def set_query_arguments(self, ef_or_spec, prefetch_offset=None, prefetch_lines=None) -> None:
        self._prefetch = {}
        if isinstance(ef_or_spec, dict):
            if prefetch_offset is not None or prefetch_lines is not None:
                raise ValueError("[zvec] positional prefetch cannot be combined with a query mapping")
            legacy_prefetch = {"ef", "prefetch_offset", "prefetch_lines"}
            expected = {"ef", "candidate_topk", "refine"}
            allowed = expected | {"prefetch"}
            keys = set(ef_or_spec)
            if keys == legacy_prefetch:
                ef = int(ef_or_spec["ef"])
                self._using_refiner = False
                self._candidate_topk = None
                self._label = self.interface
                self._prefetch = self._validate_prefetch(
                    {
                        "prefetch_offset": ef_or_spec["prefetch_offset"],
                        "prefetch_lines": ef_or_spec["prefetch_lines"],
                    }
                )
            elif keys not in (expected, allowed):
                raise ValueError(
                    "[zvec] query mapping must be an explicit-prefetch scalar query or contain "
                    "exactly ef, candidate_topk, refine, and optionally prefetch"
                )
            elif ef_or_spec["refine"] is not True:
                raise ValueError("[zvec] mapped query arguments require refine=true")
            else:
                ef = int(ef_or_spec["ef"])
                candidate_topk = int(ef_or_spec["candidate_topk"])
                if candidate_topk <= 0 or candidate_topk > ef:
                    raise ValueError(
                        f"[zvec] candidate_topk must be in [1, ef], got "
                        f"candidate_topk={candidate_topk}, ef={ef}"
                    )
                self._using_refiner = True
                self._candidate_topk = candidate_topk
                self._label = f"{self.interface}_refine"
                if "prefetch" in ef_or_spec:
                    raw_prefetch = ef_or_spec["prefetch"]
                    if not isinstance(raw_prefetch, dict) or set(raw_prefetch) != {"offset", "lines"}:
                        raise ValueError("[zvec] prefetch must contain exactly offset and lines")
                    self._prefetch = self._validate_prefetch(
                        {
                            "prefetch_offset": raw_prefetch["offset"],
                            "prefetch_lines": raw_prefetch["lines"],
                        }
                    )
        else:
            ef = int(ef_or_spec)
            self._using_refiner = False
            self._candidate_topk = None
            self._label = self.interface
            if (prefetch_offset is None) != (prefetch_lines is None):
                raise ValueError("[zvec] prefetch_offset and prefetch_lines must be supplied together")
            if prefetch_offset is not None:
                self._prefetch = self._validate_prefetch(
                    {
                        "prefetch_offset": prefetch_offset,
                        "prefetch_lines": prefetch_lines,
                    }
                )

        if ef <= 0:
            raise ValueError("[zvec] ef must be positive")
        self._ef = ef
        self._query_param = self._make_query_param()
        self._refresh_name()

    def prepare_query(self, v: np.ndarray, n: int) -> None:
        if self._candidate_topk is not None and self._candidate_topk <= n:
            raise ValueError(
                f"[zvec] candidate_topk ({self._candidate_topk}) must be " f"greater than benchmark top-k ({n})"
            )
        self._q = np.ascontiguousarray(v, dtype=np.float32)
        self._n = n

    def run_prepared_query(self) -> None:
        self._res = self._search(self._q, self._n)

    def get_prepared_query_results(self):
        return self._res

    def query(self, v: np.ndarray, n: int):
        return self._search(np.ascontiguousarray(v, dtype=np.float32), n)

    def _search(self, q: np.ndarray, n: int):
        raise NotImplementedError

    def done(self) -> None:
        self._raw_obj = None
        self._collection = None
        gc.collect()


class ZvecFastQueryDocIds(ZvecBase):
    """Direct doc-id bypass retained as a control adapter."""

    interface = "fast_query_doc_ids"

    def __init__(self, metric: str, dim: int, method_param: dict):
        super().__init__(metric, dim, method_param)
        self._label = "fast_query_doc_ids_only"
        self.name = f"zvec-{self._label}({method_param})"

    def set_query_arguments(self, ef_or_spec, prefetch_offset=None, prefetch_lines=None) -> None:
        super().set_query_arguments(ef_or_spec, prefetch_offset, prefetch_lines)
        if self._using_refiner:
            self._label = "fast_query_doc_ids_refine"
            self._search = self._search_refine
        else:
            self._label = "fast_query_doc_ids_only"
            self.__dict__.pop("_search", None)
        self._refresh_name()

    def _search(self, q: np.ndarray, n: int):
        return self._raw_obj.fast_query_doc_ids_only(VECTOR_FIELD, q, n, self._query_param)

    def _search_refine(self, q: np.ndarray, n: int):
        ids, _scores = self._raw_obj.fast_query_doc_ids(
            VECTOR_FIELD,
            q,
            self._candidate_topk,
            self._query_param,
        )
        return ids[:n]


class ZvecQuery(ZvecBase):
    """Public Collection.query adapter retained for config-sift.yml."""

    interface = "query"

    def _search(self, q: np.ndarray, n: int):
        results = self._collection.query(
            queries=Query(
                field_name=VECTOR_FIELD,
                vector=q,
                param=self._query_param,
            ),
            topk=n,
            output_fields=[],
        )
        if results is None:
            return []
        return [int(doc.id) for doc in results]


class ZvecAnnBenchDocIds(ZvecFastQueryDocIds):
    """Cached-indexer ann-benchmarks hot path used by the active config."""

    interface = "ann_bench_doc_ids"

    def __init__(self, metric: str, dim: int, method_param: dict):
        super().__init__(metric, dim, method_param)
        self._label = self.interface
        self.name = f"zvec-{self._label}({method_param})"

    def _prepare_ann_bench(self) -> None:
        self._raw_obj.ann_bench_prepare(VECTOR_FIELD)
        self._out_buf = np.empty(10, dtype=np.int64)

    def fit(self, X: np.ndarray) -> None:
        super().fit(X)
        self._prepare_ann_bench()

    def set_query_arguments(self, ef_or_spec, prefetch_offset=None, prefetch_lines=None) -> None:
        # Call the shared parser directly so the direct adapter does not bind
        # its fast_query_doc_ids implementation onto this instance.
        ZvecBase.set_query_arguments(self, ef_or_spec, prefetch_offset, prefetch_lines)
        self._raw_obj.ann_bench_set_query_params(self._query_param)
        if self._using_refiner:
            self._label = "ann_bench_doc_ids_refine"
            self._search = self._search_refine
        else:
            self._label = "ann_bench_doc_ids"
            self.__dict__.pop("_search", None)
        self._refresh_name()

    def prepare_query(self, v: np.ndarray, n: int) -> None:
        super().prepare_query(v, n)
        if len(self._out_buf) != n:
            self._out_buf = np.empty(n, dtype=np.int64)

    def _search(self, q: np.ndarray, n: int):
        self._raw_obj.ann_bench_search_fast(q, self._out_buf)
        return self._out_buf

    def _search_refine(self, q: np.ndarray, n: int):
        self._raw_obj.ann_bench_search_fast(
            q,
            self._out_buf,
            self._candidate_topk,
        )
        return self._out_buf
