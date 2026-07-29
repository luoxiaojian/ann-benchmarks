"""Refine-capable copy of the standard ann-benchmarks adapters for zvec.

Three adapters are provided, one per zvec search entry point, so the
recall/QPS trade-off of each interface can be compared directly:

* ``ZvecQuery``           -> ``Collection.query`` (full SQL/Arrow pipeline).
* ``ZvecFastQuery``       -> ``Collection.fast_query`` (bypass, string ids).
* ``ZvecFastQueryDocIds`` -> ``Collection.fast_query_doc_ids_only`` /
  ``fast_query_doc_ids`` (cheapest bypass, returns internal int64 doc ids).
* ``ZvecAnnBenchDocIds``  -> ``Collection.ann_bench_search_doc_ids_only``
  (ann-benchmarks bypass: cached indexers, params set once like qsgngt).

All three share the same fit step: vectors are inserted in row order so the
internal doc id equals the dataset row index (the contract ann-benchmarks
relies on, since the runner indexes ``X_train[idx]``). Only the search itself
is timed via the 3-stage ``prepare_query`` / ``run_prepared_query`` /
``get_prepared_query_results`` protocol, matching the high-scoring leaderboard
bindings.

Both HNSW and Vamana index families are supported (selected via the
``index`` method param), with the build/search parameters and int8
quantization defaults taken from the tuned sift/gist workspace configs.
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
    HnswQueryParam,
    LogLevel,
    OptimizeOption,
    Query,
    VamanaQueryParam,
    VectorSchema,
    create_and_open,
    open as zvec_open,
)
from zvec.model.param import HnswIndexParam, VamanaIndexParam
from zvec.typing import DataType, MetricType, QuantizeType

from ..base.module import BaseANN

# zvec must be initialized exactly once per process; ann-benchmarks may
# instantiate several adapters in the same container.
#
# ann-benchmarks pins each Docker container to a single CPU via --cpuset-cpus.
# Zvec's cgroup detection currently reads CPU quota rather than cpuset, so its
# defaults can still reflect the host CPU count.  Explicitly use one unbound
# optimize worker and disable query thread binding to match the container's
# actual CPU allocation.
try:
    zvec.init(
        log_level=LogLevel.WARN,
        # query_threads=1,
        query_thread_binding=False,
        optimize_threads=1,
        optimize_thread_binding=False,
    )
except RuntimeError:
    pass

VECTOR_FIELD = "vector"

_METRIC = {
    "euclidean": MetricType.L2,
    "angular": MetricType.COSINE,
}

_QUANTIZE = {
    "none": QuantizeType.UNDEFINED,
    "fp16": QuantizeType.FP16,
    "int8": QuantizeType.INT8,
    "uniform_int8": QuantizeType.UNIFORM_INT8,
    "uniform_uint8": QuantizeType.UNIFORM_UINT8,
}

# Workspace defaults: HNSW pairs with UniformInt8, Vamana with Int8.
_DEFAULT_QUANTIZE = {"hnsw": "uniform_int8", "vamana": "int8"}

# Tuned prefetch per index geometry (sift/gist workspace); applied when query
# args are a plain ef scalar (vsag-style config) rather than explicit dicts.
_HNSW_PREFETCH = {
    16: (32, 0),
    24: (48, 0),
    32: (64, 0),
    48: (96, 2),
}
_VAMANA_PREFETCH = {
    32: (32, 4),
    48: (48, 2),
    64: (64, 2),
}

_COUPLED_ARGS_KEY = "coupled_args"


def _flatten_coupled_args(method_param: dict) -> dict:
    """Flatten one zvec-specific coupled build-parameter bundle.

    ann-benchmarks treats the list assigned to ``coupled_args`` as one normal
    Cartesian axis. Each generated definition therefore reaches this adapter
    with a single dictionary under that key. Flatten it here so no generic
    ann-benchmarks framework changes are required.
    """
    normalized = dict(method_param)
    coupled_args = normalized.pop(_COUPLED_ARGS_KEY, None)
    if coupled_args is None:
        return normalized
    if not isinstance(coupled_args, dict):
        raise ValueError(
            "[zvec] coupled_args must resolve to one dictionary; "
            "configure it as a non-empty list of dictionaries"
        )
    conflicts = normalized.keys() & coupled_args.keys()
    if conflicts:
        raise ValueError(
            "[zvec] duplicated method parameters between regular arguments "
            f"and coupled_args: {sorted(conflicts)}"
        )
    normalized.update(coupled_args)
    return normalized


class ZvecBase(BaseANN):
    """Shared fit + 3-stage query plumbing; subclasses pick the search path."""

    interface = "base"

    def __init__(self, metric: str, dim: int, method_param: dict):
        method_param = _flatten_coupled_args(method_param)
        if metric not in _METRIC:
            raise ValueError(f"[zvec] unsupported metric: {metric}")
        self._metric_name = metric
        self._metric = _METRIC[metric]
        self._dim = int(dim)
        self._method_param = method_param

        self._index_type = str(method_param.get("index", "hnsw")).lower()
        if self._index_type not in ("hnsw", "vamana"):
            raise ValueError(f"[zvec] unsupported index: {self._index_type}")

        # HNSW build params (workspace: m in {16,24,32,48,64}, efc default).
        self._m = int(method_param.get("M", 32))
        self._ef_construction = int(
            method_param.get("efConstruction", method_param.get("ef_construction", 500))
        )
        # Vamana build params (workspace: max_degree in {16,...,64},
        # search_list_size=500, alpha=1.5).
        self._max_degree = int(method_param.get("max_degree", 32))
        self._search_list_size = int(method_param.get("search_list_size", 500))
        self._alpha = float(method_param.get("alpha", 1.5))
        self._reverse_prune_batch_size = int(
            method_param.get("reverse_prune_batch_size", 1)
        )
        self._two_pass_build = bool(
            method_param.get(
                "two_pass_build",
                method_param.get("two_pass_build_enable", False),
            )
        )

        quantize = str(
            method_param.get("quantize", _DEFAULT_QUANTIZE[self._index_type])
        ).lower()
        if quantize not in _QUANTIZE:
            raise ValueError(f"[zvec] unsupported quantize: {quantize}")
        self._quantize_name = quantize
        self._quantize = _QUANTIZE[quantize]
        self._use_contiguous_memory = bool(
            method_param.get("use_contiguous_memory", True)
        )
        self._use_flat_contiguous_memory = bool(
            method_param.get("use_flat_contiguous_memory", False)
        )

        # Search-time prefetch (QueryParam); set via set_query_arguments, not build args.
        self._prefetch: dict[str, int] = {}

        self._ef = self._search_list_size
        self._query_param = self._make_query_param(self._ef)
        self._label = self.interface
        self._collection = None
        self._path = os.path.join(
            "zvec_indices",
            f"{self.interface}_{self._index_type}_{metric}_d{self._dim}"
            f"_{self._build_tag()}_{quantize}",
        )
        self.name = f"zvec-{self.interface}({method_param})"

        # Working buffers for the timed query path.
        self._q = None
        self._n = 0
        self._res = None

    def _build_tag(self) -> str:
        memory_tag = (
            f"cm{int(self._use_contiguous_memory)}"
            f"_fcm{int(self._use_flat_contiguous_memory)}"
        )
        if self._index_type == "vamana":
            passes = "2pass" if self._two_pass_build else "1pass"
            return (
                f"R{self._max_degree}_L{self._search_list_size}"
                f"_B{self._reverse_prune_batch_size}"
                f"_a{self._alpha}_{passes}_{memory_tag}"
            )
        return f"m{self._m}_efc{self._ef_construction}_{memory_tag}"

    def _make_index_param(self):
        if self._index_type == "vamana":
            return VamanaIndexParam(
                metric_type=self._metric,
                max_degree=self._max_degree,
                search_list_size=self._search_list_size,
                alpha=self._alpha,
                reverse_prune_batch_size=self._reverse_prune_batch_size,
                use_contiguous_memory=self._use_contiguous_memory,
                two_pass_build=self._two_pass_build,
                quantize_type=self._quantize,
                use_flat_contiguous_memory=self._use_flat_contiguous_memory,
            )
        return HnswIndexParam(
            metric_type=self._metric,
            m=self._m,
            ef_construction=self._ef_construction,
            use_contiguous_memory=self._use_contiguous_memory,
            quantize_type=self._quantize,
            use_flat_contiguous_memory=self._use_flat_contiguous_memory,
        )

    def _make_query_param(self, ef: int):
        extra = dict(self._prefetch) if self._prefetch else {}
        if self._index_type == "vamana":
            return VamanaQueryParam(ef_search=int(ef), extra_params=extra)
        return HnswQueryParam(ef=int(ef), extra_params=extra)

    def _default_prefetch(self) -> dict[str, int]:
        if self._index_type == "vamana":
            po, pl = _VAMANA_PREFETCH.get(self._max_degree, (0, 0))
        else:
            po, pl = _HNSW_PREFETCH.get(self._m, (0, 0))
        return {"prefetch_offset": po, "prefetch_lines": pl}

    def _parse_query_spec(
        self, ef_or_spec, prefetch_offset=None, prefetch_lines=None
    ) -> tuple[int, dict[str, int]]:
        if isinstance(ef_or_spec, dict):
            spec = ef_or_spec
            ef = int(spec["ef"])
            if "prefetch" in spec:
                if "prefetch_offset" in spec or "prefetch_lines" in spec:
                    raise ValueError(
                        "[zvec] prefetch cannot be combined with "
                        "prefetch_offset/prefetch_lines"
                    )
                prefetch = spec["prefetch"]
                if not isinstance(prefetch, dict):
                    raise ValueError(
                        "[zvec] prefetch must be a mapping with offset and lines"
                    )
                if set(prefetch) != {"offset", "lines"}:
                    raise ValueError(
                        "[zvec] prefetch must contain exactly offset and lines"
                    )
                po = int(prefetch["offset"])
                pl = int(prefetch["lines"])
            else:
                defaults = self._default_prefetch()
                po = int(
                    spec.get("prefetch_offset", defaults["prefetch_offset"])
                )
                pl = int(
                    spec.get("prefetch_lines", defaults["prefetch_lines"])
                )
        else:
            ef = int(ef_or_spec)
            if prefetch_offset is None and prefetch_lines is None:
                return ef, self._default_prefetch()
            defaults = self._default_prefetch()
            po = (
                defaults["prefetch_offset"]
                if prefetch_offset is None
                else int(prefetch_offset)
            )
            pl = (
                defaults["prefetch_lines"]
                if prefetch_lines is None
                else int(prefetch_lines)
            )
        if not 0 <= po <= 256:
            raise ValueError("[zvec] prefetch offset must be in [0, 256]")
        if not 0 <= pl <= 256:
            raise ValueError("[zvec] prefetch lines must be in [0, 256]")
        return ef, {"prefetch_offset": po, "prefetch_lines": pl}

    # --- fit -----------------------------------------------------------------
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
        batch_size = 1024
        total = len(X)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            docs = [
                Doc(id=str(i), vectors={VECTOR_FIELD: X[i].tolist()})
                for i in range(start, end)
            ]
            results = build_col.insert(docs)
            if not isinstance(results, list):
                results = [results]
            for result in results:
                if not result.ok():
                    raise RuntimeError(f"[zvec] insert failed: {result.code()}")
        build_col.optimize(option=OptimizeOption())

        # Release the single-writer handle so we can reopen read-only (mmap),
        # the path the bench measures search on.
        build_col = None
        gc.collect()
        self._collection = zvec_open(
            self._path, CollectionOption(read_only=True, enable_mmap=True)
        )
        # Cache the raw C++ pybind11 object to bypass the Python
        # Collection → QueryExecutor middleware on every query.
        # This matches what zvec/bench does (client.collection._obj).
        self._raw_obj = self._collection._obj

    def set_query_arguments(self, ef_or_spec, prefetch_offset=None, prefetch_lines=None) -> None:
        self._ef, self._prefetch = self._parse_query_spec(
            ef_or_spec, prefetch_offset, prefetch_lines
        )
        self._query_param = self._make_query_param(self._ef)
        qtag = dict(self._prefetch) if self._prefetch else {}
        self.name = (
            f"zvec-{self._label}({self._method_param}, ef={self._ef}"
            + (f", {qtag}" if qtag else "")
            + ")"
        )

    # --- 3-stage protocol: only run_prepared_query() is timed ----------------
    def prepare_query(self, v: np.ndarray, n: int) -> None:
        self._q = v
        self._n = n

    def run_prepared_query(self) -> None:
        self._res = self._search(self._q, self._n)

    def get_prepared_query_results(self):
        return self._res

    # --- non-prepared path (kept for completeness) ---------------------------
    def query(self, v: np.ndarray, n: int):
        return self._search(np.ascontiguousarray(v, dtype=np.float32), n)

    def _search(self, q: np.ndarray, n: int):
        raise NotImplementedError

    def done(self) -> None:
        self._raw_obj = None
        self._collection = None
        gc.collect()


class ZvecQuery(ZvecBase):
    """Full pipeline path: ``Collection.query`` -> primary-key ids."""

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


class ZvecFastQuery(ZvecBase):
    """Bypass path: ``Collection.fast_query`` -> primary-key ids."""

    interface = "fast_query"

    def _search(self, q: np.ndarray, n: int):
        ids, _scores = self._raw_obj.fast_query(
            VECTOR_FIELD, q, n, self._query_param
        )
        return [int(x) for x in ids]


class ZvecFastQueryDocIds(ZvecBase):
    """Cheapest bypass: ``fast_query_doc_ids_only`` -> internal int64 doc ids.

    With ``with_scores`` set, uses ``fast_query_doc_ids`` (ids + scores) instead.
    A query dictionary containing ``refine: true``, ``candidate_topk``, or
    ``candidates`` requests a larger candidate set through
    ``fast_query_doc_ids`` and returns top-k ids after FP32 reranking.
    The internal doc id equals the dataset row index because vectors are
    inserted in row order during fit.
    """

    interface = "fast_query_doc_ids"

    def __init__(self, metric: str, dim: int, method_param: dict):
        self._using_refiner = False
        self._candidate_topk = None
        super().__init__(metric, dim, method_param)
        self._with_scores = bool(method_param.get("with_scores", False))
        self._label = (
            "fast_query_doc_ids"
            if self._with_scores
            else "fast_query_doc_ids_only"
        )
        self.name = f"zvec-{self._label}({method_param})"

    def _make_query_param(self, ef: int):
        if not self._using_refiner:
            return super()._make_query_param(ef)
        extra = dict(self._prefetch) if self._prefetch else {}
        if self._index_type == "vamana":
            return VamanaQueryParam(
                ef_search=int(ef),
                is_using_refiner=True,
                extra_params=extra,
            )
        return HnswQueryParam(
            ef=int(ef),
            is_using_refiner=True,
            extra_params=extra,
        )

    def set_query_arguments(
        self, ef_or_spec, prefetch_offset=None, prefetch_lines=None
    ) -> None:
        is_spec = isinstance(ef_or_spec, dict)
        using_refiner = is_spec and (
            bool(ef_or_spec.get("refine", False))
            or "candidate_topk" in ef_or_spec
            or "candidates" in ef_or_spec
        )
        if not using_refiner:
            self._using_refiner = False
            self._candidate_topk = None
            self._label = (
                "fast_query_doc_ids"
                if self._with_scores
                else "fast_query_doc_ids_only"
            )
            super().set_query_arguments(
                ef_or_spec,
                prefetch_offset=prefetch_offset,
                prefetch_lines=prefetch_lines,
            )
            return

        candidate_values = {
            int(ef_or_spec[key])
            for key in ("candidate_topk", "candidates")
            if key in ef_or_spec
        }
        if not candidate_values:
            raise ValueError(
                "[zvec-refine] refine queries require candidate_topk or candidates"
            )
        if len(candidate_values) != 1:
            raise ValueError(
                "[zvec-refine] candidate_topk and candidates must be equal"
            )
        candidate_topk = candidate_values.pop()
        ef, prefetch = self._parse_query_spec(ef_or_spec)
        if candidate_topk <= 0:
            raise ValueError("[zvec-refine] candidate_topk must be positive")
        if candidate_topk > ef:
            raise ValueError(
                f"[zvec-refine] candidate_topk ({candidate_topk}) cannot "
                f"exceed ef ({ef})"
            )

        self._using_refiner = True
        self._ef = ef
        self._candidate_topk = candidate_topk
        self._prefetch = prefetch
        self._query_param = self._make_query_param(ef)
        self._label = "fast_query_doc_ids_refine"
        self.name = (
            f"zvec-{self._label}({self._method_param}, ef={ef}, "
            f"candidate_topk={candidate_topk}, prefetch={prefetch})"
        )

    def _search(self, q: np.ndarray, n: int):
        if self._using_refiner:
            return self._search_refine(q, n)
        if self._with_scores:
            ids, _scores = self._raw_obj.fast_query_doc_ids(
                VECTOR_FIELD, q, n, self._query_param
            )
            return ids
        return self._raw_obj.fast_query_doc_ids_only(
            VECTOR_FIELD, q, n, self._query_param
        )

    def _search_refine(self, q: np.ndarray, n: int):
        if self._candidate_topk is None:
            raise RuntimeError(
                "[zvec-refine] set_query_arguments must be called before querying"
            )
        if self._candidate_topk <= n:
            raise ValueError(
                f"[zvec-refine] candidate_topk ({self._candidate_topk}) must "
                f"be greater than benchmark top-k ({n})"
            )
        ids, _scores = self._raw_obj.fast_query_doc_ids(
            VECTOR_FIELD,
            q,
            self._candidate_topk,
            self._query_param,
        )
        return ids[:n]

    # --- batch query support (used with --batch flag) -------------------------
    def prepare_batch_query(self, X, n):
        self._batch_X = np.ascontiguousarray(X, dtype=np.float32)
        self._batch_n = n

    def run_batch_query(self):
        if self._using_refiner:
            self._batch_res = [
                self._search(query, self._batch_n) for query in self._batch_X
            ]
        else:
            self._batch_res = self._raw_obj.batch_fast_query_doc_ids_only(
                VECTOR_FIELD, self._batch_X, self._batch_n,
                self._query_param
            )

    def get_batch_results(self):
        return [self._batch_res[i] for i in range(len(self._batch_res))]


class ZvecAnnBenchDocIds(ZvecFastQueryDocIds):
    """Ann-benchmarks bypass: cached C++ indexers + params set once.

    Uses ``ann_bench_prepare`` / ``ann_bench_set_query_params`` /
    ``ann_bench_search_fast`` on the raw collection object. Refine queries use
    the same cached-indexer and preallocated-output path; the native search
    gathers the coarse candidates and reranks them through the reference Flat
    index. Non-refine queries retain the original branch-free ``_search`` hot
    path.
    """

    interface = "ann_bench_doc_ids"

    def __init__(self, metric: str, dim: int, method_param: dict):
        super().__init__(metric, dim, method_param)
        self._label = "ann_bench_doc_ids"
        self.name = f"zvec-{self._label}({method_param})"

    def fit(self, X: np.ndarray) -> None:
        super().fit(X)
        self._raw_obj.ann_bench_prepare(VECTOR_FIELD)
        # Pre-allocate output buffer (count is always fixed during a run).
        self._out_buf = np.empty(10, dtype=np.int64)

    def set_query_arguments(self, ef_or_spec, prefetch_offset=None, prefetch_lines=None) -> None:
        super().set_query_arguments(ef_or_spec, prefetch_offset, prefetch_lines)
        # Bind the mode once when query arguments change. The non-refine hot
        # path therefore remains the original branch-free class method.
        if self._using_refiner:
            self._search = self._search_refine
        else:
            self.__dict__.pop("_search", None)
        self._label = (
            "ann_bench_doc_ids_refine"
            if self._using_refiner
            else "ann_bench_doc_ids"
        )
        self.name = (
            f"zvec-{self._label}({self._method_param}, ef={self._ef}, "
            f"candidate_topk={self._candidate_topk}, prefetch={self._prefetch})"
        )
        self._raw_obj.ann_bench_set_query_params(self._query_param)

    def _search_refine(self, q: np.ndarray, n: int):
        if self._candidate_topk is None:
            raise RuntimeError(
                "[zvec-refine] set_query_arguments must be called before querying"
            )
        if self._candidate_topk <= n:
            raise ValueError(
                f"[zvec-refine] candidate_topk ({self._candidate_topk}) must "
                f"be greater than benchmark top-k ({n})"
            )
        if (
            not hasattr(self, "_refine_out_buf")
            or len(self._refine_out_buf) != self._candidate_topk
        ):
            self._refine_out_buf = np.empty(self._candidate_topk, dtype=np.int64)
        self._raw_obj.ann_bench_search_fast(q, self._refine_out_buf)
        return self._refine_out_buf[:n]

    def _search(self, q: np.ndarray, n: int):
        if len(self._out_buf) != n:
            self._out_buf = np.empty(n, dtype=np.int64)
        self._raw_obj.ann_bench_search_fast(q, self._out_buf)
        return self._out_buf
