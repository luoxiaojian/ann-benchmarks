"""Integration checks for the two prepared-native zvec 0.8.0 adapters.

These tests build disposable indexes and compare the prepared C++ calls with
the corresponding public Python APIs. They do not measure performance.
"""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import yaml

zvec = pytest.importorskip("zvec")
pytestmark = pytest.mark.skipif(version("zvec") != "0.8.0", reason="requires zvec==0.8.0")

from ann_benchmarks.algorithms.zvec import module as adapter
from ann_benchmarks.definitions import create_definitions_from_algorithm
from ann_benchmarks.results import build_result_filepath, store_results


CONFIG_PATH = Path(__file__).resolve().parents[1] / "ann_benchmarks/algorithms/zvec/config.yml"


def _params(**overrides):
    return {"max_degree": 16, "search_list_size": 48, "alpha": 1.3, **overrides}


def _prepared_ids(algo, vector, topk):
    algo.prepare_query(vector, topk)
    algo.run_prepared_query()
    return list(algo.get_prepared_query_results())


@pytest.mark.parametrize(
    "quantize,flat_type,dimension,count,query_args",
    [
        ("uniform_uint7", "inherit", 128, 1536, [17, 40]),
        (
            "uniform_uint4",
            "uint8",
            128,
            1536,
            [{"ef_search": 40, "scale_factor": 2.0}, {"ef_search": 48, "scale_factor": 3.8}],
        ),
        ("int8", "fp16", 960, 192, [40, {"ef_search": 48, "scale_factor": 1.4}]),
    ],
    ids=["uniform-uint7-graph", "uniform-uint4-uint8-refine-graph", "record-int8-fp16"],
)
def test_prepared_paths_match_public_apis_and_training_rows(
    tmp_path, monkeypatch, quantize, flat_type, dimension, count, query_args
):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(20261022)
    if quantize == "int8":
        train = rng.normal(size=(count, dimension)).astype(np.float32)
    else:
        train = rng.integers(0, 180, size=(count, dimension)).astype(np.float32)
    params = _params(
        quantize=quantize,
        flat_data_type=flat_type,
        use_flat_contiguous_memory=flat_type != "inherit",
    )

    # Build exactly one graph for both adapters, using the normal fit pipeline.
    builder = adapter.ZvecQuery("euclidean", dimension, params)
    try:
        builder.fit(train)
        # Numeric internal IDs must still address the original ANN training
        # rows after insertion in multiple batches, optimize and readonly reopen.
        resolved = builder._collection.resolve_internal_ids(np.arange(count, dtype=np.int64))
        assert resolved == [str(i) for i in range(count)]
    finally:
        builder.done()

    reference = {}
    for algo_type in (adapter.ZvecQuery, adapter.ZvecQueryInternalIds):
        algo = algo_type("euclidean", dimension, params)
        # Production paths are intentionally isolated by API. Reopen the same
        # disposable graph here so independent graph construction cannot explain
        # any difference between the two calls.
        algo._path = builder._path
        try:
            algo._open_readonly()
            for arg_index, query_arg in enumerate(query_args):
                algo.set_query_arguments(query_arg)
                for row in (0, 17, count // 2, count - 1):
                    vector = train[row]
                    prepared = _prepared_ids(algo, vector, 10)
                    public_query = zvec.Query("vector", vector=vector, param=algo._query_param)
                    documents = algo._collection.query(
                        public_query, topk=10, output_fields=[], include_vector=False
                    )
                    public_primary_ids = [int(doc.id) for doc in documents if doc is not None]
                    public_internal_ids = algo._collection.query_internal_ids(public_query, topk=10)
                    internal_ids = public_internal_ids[public_internal_ids >= 0].tolist()
                    resolved = algo._collection.resolve_internal_ids(public_internal_ids)
                    assert [int(key) for key in resolved if key is not None] == internal_ids
                    assert prepared == public_primary_ids == internal_ids
                    assert len(prepared) == len(set(prepared)) == 10
                    assert all(0 <= item < count for item in prepared)
                    key = (arg_index, row)
                    if key in reference:
                        assert prepared == reference[key]
                    else:
                        reference[key] = prepared
        finally:
            algo.done()


@pytest.mark.parametrize("class_name", ["ZvecQuery", "ZvecQueryInternalIds"])
def test_topk_larger_than_collection_excludes_padding(tmp_path, monkeypatch, class_name):
    monkeypatch.chdir(tmp_path)
    train = np.random.default_rng(7).integers(0, 100, size=(5, 128)).astype(np.float32)
    algo = getattr(adapter, class_name)("euclidean", 128, _params(quantize="uniform_uint7"))
    try:
        algo.fit(train)
        algo.set_query_arguments(17)
        ids = _prepared_ids(algo, train[0], 10)
        public_query = zvec.Query("vector", vector=train[0], param=algo._query_param)
        raw_ids = algo._collection.query_internal_ids(public_query, topk=10)
        assert raw_ids.dtype == np.int64
        assert raw_ids.shape == (10,)
        assert np.count_nonzero(raw_ids == -1) == 5
        assert ids == raw_ids[raw_ids >= 0].tolist()
        assert sorted(ids) == list(range(5))
    finally:
        algo.done()


@pytest.mark.parametrize("bad_value", [0.25, -1.0, 256.0, np.nan, np.inf])
def test_incompatible_uint8_flat_exits_cleanly_before_build(tmp_path, monkeypatch, bad_value):
    monkeypatch.chdir(tmp_path)
    train = np.zeros((4, 128), dtype=np.float32)
    train[0, 0] = bad_value
    algo = adapter.ZvecQuery(
        "euclidean", 128, _params(quantize="uniform_uint4", flat_data_type="uint8")
    )
    try:
        with pytest.raises(SystemExit) as error:
            algo.fit(train)
        assert error.value.code == 0
        assert not Path(algo._path).exists()
    finally:
        algo.done()


def test_definitions_share_grid_and_store_separate_curves(tmp_path, monkeypatch):
    configurations = yaml.safe_load(CONFIG_PATH.read_text())["float"]["euclidean"]
    by_name = {configuration["name"]: configuration for configuration in configurations}
    assert set(by_name) == {"zvec-query", "zvec-query_internal_ids"}
    query_config = by_name["zvec-query"]
    internal_config = by_name["zvec-query_internal_ids"]
    assert query_config["constructor"] == "ZvecQuery"
    assert internal_config["constructor"] == "ZvecQueryInternalIds"
    assert query_config["run_groups"] == internal_config["run_groups"]

    definitions = [
        create_definitions_from_algorithm(name, by_name[name], dimension=128, count=10)
        for name in ("zvec-query", "zvec-query_internal_ids")
    ]
    assert len(definitions[0]) == len(definitions[1]) > 0
    paths = [set(), set()]
    for left, right in zip(*definitions):
        assert left.arguments == right.arguments
        assert left.query_argument_groups == right.query_argument_groups
        for side, definition in enumerate((left, right)):
            for query_arguments in definition.query_argument_groups:
                path = build_result_filepath("sift-128-euclidean", 10, definition, query_arguments)
                assert len(Path(path).name.encode()) <= 255
                assert path not in paths[side]
                paths[side].add(path)
    assert paths[0].isdisjoint(paths[1])

    # Exercise the stock writer as well as its path builder: identical training
    # and query parameters must not overwrite the other API's HDF5 output.
    monkeypatch.chdir(tmp_path)
    for side, algorithm_definitions in enumerate(definitions):
        definition = algorithm_definitions[0]
        query_arguments = definition.query_argument_groups[0]
        store_results(
            "sift-128-euclidean", 10, definition, query_arguments,
            {"algo": definition.algorithm}, [(0.001, [(side, 0.0)])], False,
        )
    import h5py

    for side, algorithm_definitions in enumerate(definitions):
        definition = algorithm_definitions[0]
        query_arguments = definition.query_argument_groups[0]
        filename = build_result_filepath("sift-128-euclidean", 10, definition, query_arguments)
        with h5py.File(filename, "r") as result:
            assert result.attrs["algo"] == definition.algorithm
            assert result["neighbors"][0, 0] == side
