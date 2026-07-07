FROM ann-benchmarks

WORKDIR /home/app

RUN lscpu

# zvec is not published with the bypass search APIs yet (fast_query /
# fast_query_doc_ids[_only]), so install the wheel built locally from the
# ~/py/zvec source tree instead of PyPI. The wheel is produced into
# ann_benchmarks/algorithms/zvec/wheels/ (see the repo README / setup) and is
# part of the Docker build context.
COPY ann_benchmarks/algorithms/zvec/wheels/*.whl /tmp/zvec_wheels/

RUN pip3 install /tmp/zvec_wheels/*.whl

RUN python3 -c 'import zvec; from _zvec import _Collection; \
    assert all(hasattr(_Collection, m) for m in ("fast_query", "fast_query_doc_ids", "fast_query_doc_ids_only", "ann_bench_prepare", "ann_bench_set_query_params", "ann_bench_search_doc_ids_only")), "zvec wheel missing bypass APIs"; \
    print("zvec", zvec.__version__, "with bypass APIs OK")'
