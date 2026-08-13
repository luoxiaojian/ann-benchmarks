FROM ann-benchmarks

WORKDIR /home/app

LABEL zvec.commit="7dce1b13237ebd4570f200689fba9a2d7d0b84fa" \
      zvec.base_commit="cdd7940de89d6fce87dbe99e8bf90f166ac1aa8d" \
      zvec.variant="runtime-vamana-build-switch" \
      zvec.vamana_build_pipeline="runtime-use_bulk_build" \
      zvec.benchmark_use_bulk_build="false" \
      zvec.uniform_uint8_query_distance="exact"

RUN lscpu

# zvec is not published with the bypass search APIs yet (fast_query /
# fast_query_doc_ids[_only]), so install the wheel built locally from the
# ~/py/zvec source tree instead of PyPI. The wheel is produced into
# ann_benchmarks/algorithms/zvec/wheels/ (see the repo README / setup) and is
# part of the Docker build context.
COPY ann_benchmarks/algorithms/zvec/zvec-0.5.1.dev44-cp310-cp310-linux_x86_64.whl /tmp/zvec_wheels/

RUN pip3 install /tmp/zvec_wheels/*.whl

RUN python3 -c 'import zvec; from _zvec import _Collection; from zvec.typing import DataType, QuantizeType; \
    assert all(hasattr(_Collection, m) for m in ("fast_query", "fast_query_doc_ids", "fast_query_doc_ids_only", "ann_bench_prepare", "ann_bench_set_query_params", "ann_bench_search_doc_ids_only", "ann_bench_search_fast")), "zvec wheel missing bypass APIs"; \
    assert zvec.VamanaIndexParam(use_bulk_build=False).use_bulk_build is False, "zvec wheel missing runtime Vamana build switch"; \
    assert zvec.VamanaIndexParam(use_bulk_build=True).use_bulk_build is True, "zvec wheel cannot enable optimized Vamana build"; \
    assert hasattr(QuantizeType, "UNIFORM_UINT4"), "zvec wheel missing UniformUint4"; \
    assert hasattr(DataType, "VECTOR_UINT8"), "zvec wheel missing raw uint8 Flat storage"; \
    print("zvec", zvec.__version__, "with runtime Vamana build switch, bypass, UniformUint4, and raw uint8 Flat APIs OK")'
