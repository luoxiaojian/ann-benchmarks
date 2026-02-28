"""
Zvec integration for ann-benchmarks.
Reference implementation from VectorDBBench: https://github.com/zilliztech/VectorDBBench
Zvec is Alibaba's production-grade in-process vector database built on Proxima.
"""
import os
import shutil
import tempfile

import zvec
from zvec import CollectionOption, CollectionSchema, DataType, Doc, OptimizeOption, VectorQuery, VectorSchema
from zvec.model.param import HnswIndexParam, HnswQueryParam

from ..base.module import BaseANN


class Zvec(BaseANN):
    """
    Zvec HNSW implementation for ann-benchmarks.
    Based on VectorDBBench's proven configuration for optimal performance.
    """

    def __init__(self, metric, index_params):
        self._metric = metric
        self._index_params = index_params
        
        # HNSW index parameters - following VectorDBBench defaults
        self._m = int(index_params.get("M", 50))
        self._ef_construction = int(index_params.get("efConstruction", 500))
        
        # Query-time parameter (set later)
        self._ef = None
        
        self._collection = None
        self._tempdir = None
        self._path = None
        
        # Map metric to zvec MetricType
        self._metric_type = self._translate_metric(metric)

    @staticmethod
    def _translate_metric(metric):
        """Translate ANN benchmark metric to zvec MetricType"""
        metric_map = {
            "angular": zvec.MetricType.COSINE,
            "euclidean": zvec.MetricType.L2,
            "ip": zvec.MetricType.IP,
        }
        if metric not in metric_map:
            raise ValueError(f"Unsupported metric: {metric}")
        return metric_map[metric]

    def fit(self, X):
        """Build the HNSW index from training data"""
        # Create temporary directory for zvec collection
        self._tempdir = tempfile.mkdtemp(prefix="zvec_hnsw_")
        self._path = os.path.join(self._tempdir, "collection")
        
        # Get dimension from data
        dimension = X.shape[1]
        
        # Create schema with HNSW index parameters
        # Following VectorDBBench configuration
        hnsw_params = HnswIndexParam(
            metric_type=self._metric_type,
            m=self._m,
            ef_construction=self._ef_construction,
        )
        
        schema = CollectionSchema(
            name="ann_benchmark_hnsw",
            vectors=VectorSchema(
                name="dense",
                data_type=DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=hnsw_params,
            ),
        )
        
        # Create collection with mmap enabled for better performance
        option = CollectionOption(read_only=False, enable_mmap=True)
        self._collection = zvec.create_and_open(path=self._path, schema=schema, option=option)
        
        # Batch insert vectors as documents
        docs = [
            Doc(
                id=str(idx),
                vectors={"dense": vector.tolist() if hasattr(vector, "tolist") else list(vector)},
            )
            for idx, vector in enumerate(X)
        ]
        
        self._collection.insert(docs)
        
        # Optimize after insertion (following VectorDBBench practice)
        self._collection.optimize(option=OptimizeOption())

    def set_query_arguments(self, ef):
        """Set query-time parameters"""
        self._ef = int(ef)
        self.name = f"Zvec(M={self._m}, efConstruction={self._ef_construction}, ef={self._ef})"

    def query(self, v, n):
        """Query for nearest neighbors"""
        if self._ef is None:
            raise ValueError("Query arguments not set. Call set_query_arguments() first.")
        
        # Create query parameters
        query_params = HnswQueryParam(ef=self._ef)
        
        # Perform vector query
        vector = v.tolist() if hasattr(v, "tolist") else list(v)
        results = self._collection.query(
            output_fields=[],
            topk=n,
            filter="",
            vectors=VectorQuery(field_name="dense", vector=vector, param=query_params),
        )
        
        # Extract indices from results
        return [int(result.id) for result in results]

    def done(self):
        """Cleanup resources"""
        if self._collection is not None:
            del self._collection
            self._collection = None

        if self._tempdir is not None and os.path.exists(self._tempdir):
            shutil.rmtree(self._tempdir)
            self._tempdir = None

    def __str__(self):
        return self.name
