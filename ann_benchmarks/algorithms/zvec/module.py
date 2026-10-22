"""zvec 0.8.0 Vamana benchmarks for prepared Query and QueryInternalIds."""

import os
import shutil

import numpy as np
import zvec
from zvec._zvec.param import _SearchQuery

from ..base.module import BaseANN

try:
    # ANN pins each container to one CPU; leave query-thread affinity to it.
    zvec.init(log_level=zvec.LogLevel.WARN, optimize_threads=1)
except RuntimeError:
    # Local validation may have initialized zvec already.
    pass


class _ZvecVamanaBase(BaseANN):
    _api_name = "base"

    def __init__(self, metric: str, dim: int, method_param: dict):
        params = dict(method_param)
        quantize = params.pop("quantize", "int8")
        flat_type = params.pop("flat_data_type", "inherit")
        self._index_param = zvec.VamanaIndexParam(
            metric_type={"euclidean": zvec.MetricType.L2}[metric],
            use_contiguous_memory=True,
            quantize_type=getattr(zvec.QuantizeType, quantize.upper()),
            flat_data_type={
                "inherit": zvec.DataType.VECTOR_FP32,
                "fp16": zvec.DataType.VECTOR_FP16,
                "uint8": zvec.DataType.VECTOR_UINT8,
            }[flat_type],
            **params,
        )
        self._dim = dim
        self._method_param = dict(method_param)
        self._path = os.path.join(
            "zvec_indices",
            "0.8.0",
            self._api_name,
            f"d{dim}_R{params['max_degree']}_L{params['search_list_size']}_a{params['alpha']}_{flat_type}_{quantize}",
        )
        self._collection = None
        self._search_call = None
        self._vector_schema = None
        self._prepared_native_query = None
        self._query_vector = None
        self._res = None
        self.set_query_arguments(self._index_param.search_list_size)

    def fit(self, X: np.ndarray) -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self._dim:
            raise ValueError(f"Expected training vectors with dimension {self._dim}")
        # UINT8 raw storage requires finite integers in [0, 255].
        # Skip incompatible datasets before building the index.
        if self._index_param.flat_data_type == zvec.DataType.VECTOR_UINT8:
            for start in range(0, len(X), 16384):
                chunk = X[start : start + 16384]
                if (
                    not np.isfinite(chunk).all()
                    or (chunk < 0).any()
                    or (chunk > 255).any()
                    or not np.equal(chunk, chunk.astype(np.uint8)).all()
                ):
                    print(
                        f"[zvec-{self._api_name}] Skipping incompatible data: "
                        "UINT8 raw Flat requires integer vectors in [0, 255]",
                        flush=True,
                    )
                    raise SystemExit(0)

        self.done()
        if os.path.exists(self._path):
            shutil.rmtree(self._path)
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        schema = zvec.CollectionSchema(
            name="annb_query_api",
            fields=[],
            vectors=[
                zvec.VectorSchema(
                    "vector",
                    zvec.DataType.VECTOR_FP32,
                    dimension=self._dim,
                    index_param=self._index_param,
                )
            ],
        )
        collection = zvec.create_and_open(
            path=self._path,
            schema=schema,
            option=zvec.CollectionOption(read_only=False, enable_mmap=True),
        )
        try:
            for start in range(0, len(X), 1024):
                docs = [
                    zvec.Doc(id=str(i), vectors={"vector": X[i].tolist()})
                    for i in range(start, min(start + 1024, len(X)))
                ]
                for result in collection.insert(docs):
                    if not result.ok():
                        raise RuntimeError(f"Insert failed: {result.code()}")
            collection.optimize(option=zvec.OptimizeOption())
        finally:
            collection.close()
        self._open_readonly()

    def _open_readonly(self) -> None:
        self._collection = zvec.open(self._path, zvec.CollectionOption(read_only=True, enable_mmap=True))
        self._vector_schema = self._collection.schema.vector("vector")._get_object()

    def set_query_arguments(self, query_args) -> None:
        params = dict(query_args) if isinstance(query_args, dict) else {"ef_search": query_args}
        scale_factor = params.pop("scale_factor", 1.0)
        if scale_factor != 1.0:
            params.update(is_using_refiner=True, scale_factor=scale_factor)
        self._query_param = zvec.VamanaQueryParam(**params)
        self.name = f"zvec-{self._api_name}({self._method_param}, query={query_args})"

    def prepare_query(self, v: np.ndarray, n: int) -> None:
        # Build the native request outside ANN-Benchmarks' query timer.
        query = _SearchQuery()
        query.topk = n
        query.field_name = "vector"
        query.query_params = self._query_param
        # Keep converted input alive until the prepared query has completed.
        self._query_vector = np.ascontiguousarray(v, dtype=np.float32)
        query.set_vector(self._vector_schema, self._query_vector)
        self._prepared_native_query = query

    def query(self, v: np.ndarray, n: int):
        self.prepare_query(v, n)
        self.run_prepared_query()
        return self.get_prepared_query_results()

    def done(self) -> None:
        self._search_call = None
        self._vector_schema = None
        self._prepared_native_query = None
        self._query_vector = None
        self._res = None
        if self._collection is not None:
            self._collection.close()
            self._collection = None

    def get_additional(self):
        return {
            "zvec_version": str(zvec.__version__),
            "zvec_query_api": self._api_name,
            "zvec_query_mode": "prepared-native",
        }


class ZvecQuery(_ZvecVamanaBase):
    """Benchmark the prepared native Query path."""

    _api_name = "query"

    def _open_readonly(self) -> None:
        super()._open_readonly()
        self._search_call = self._collection._obj.Query

    def run_prepared_query(self) -> None:
        self._res = self._search_call(self._prepared_native_query)

    def get_prepared_query_results(self):
        # Convert primary keys back to training-row IDs outside the query timer.
        return [int(item[0]) for item in self._res if item is not None]


class ZvecQueryInternalIds(_ZvecVamanaBase):
    """Benchmark the prepared native query_internal_ids path."""

    _api_name = "query_internal_ids"

    def _open_readonly(self) -> None:
        super()._open_readonly()
        self._search_call = self._collection._obj.QueryInternalIds

    def run_prepared_query(self) -> None:
        self._res = self._search_call(self._prepared_native_query, False)

    def get_prepared_query_results(self):
        # Omit -1 padding; ANN-Benchmarks pads results itself.
        return self._res[self._res >= 0]
