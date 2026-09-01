FROM ann-benchmarks

WORKDIR /home/app

LABEL zvec.wheel="zvec-0.0.1.dev296-cp310-cp310-linux_x86_64.whl" \
      zvec.wheel_sha256="de71d374c312b64a6ab3a024b8140f915e970293f7e923650e19abfd8b8b50d0" \
      zvec.source_branch="uniform_uint8" \
      zvec.commit="c3e594536db21dd93909f70394deb3385dda2724" \
      zvec.vamana_build_pipeline="runtime-use_bulk_build" \
      zvec.refine_flat_data_type="fp16,uint8"

RUN lscpu

# Wheel provenance: zvec-0.0.1.dev296-cp310-cp310-linux_x86_64.whl was built
# from zvec branch uniform_uint8 at commit
# c3e594536db21dd93909f70394deb3385dda2724.
# SHA-256: de71d374c312b64a6ab3a024b8140f915e970293f7e923650e19abfd8b8b50d0.
# Update the wheel, source_branch, and commit together when changing builds.
#
# zvec is not published with the bypass search APIs yet (fast_query /
# fast_query_doc_ids[_only]), so install the wheel built locally from the
# /root/zvec source tree instead of PyPI. It is part of the Docker build
# context alongside the archived prior wheel, so copy and install this build
# by its exact filename.
COPY ann_benchmarks/algorithms/zvec/zvec-0.0.1.dev296-cp310-cp310-linux_x86_64.whl /tmp/zvec_wheels/

RUN pip3 install /tmp/zvec_wheels/zvec-0.0.1.dev296-cp310-cp310-linux_x86_64.whl

RUN python3 -c 'import zvec; from _zvec import _Collection; from zvec.typing import DataType, QuantizeType; \
    assert str(zvec.__version__) == "0.0.1.dev296", "unexpected zvec version"; \
    assert all(hasattr(_Collection, m) for m in ("fast_query", "fast_query_doc_ids", "fast_query_doc_ids_only", "ann_bench_prepare", "ann_bench_set_query_params", "ann_bench_search_doc_ids_only", "ann_bench_search_fast")), "zvec wheel missing bypass APIs"; \
    assert zvec.VamanaIndexParam(use_bulk_build=False).use_bulk_build is False, "zvec wheel missing runtime Vamana build switch"; \
    assert zvec.VamanaIndexParam(use_bulk_build=True).use_bulk_build is True, "zvec wheel cannot enable optimized Vamana build"; \
    assert hasattr(QuantizeType, "UNIFORM_UINT4"), "zvec wheel missing UniformUint4"; \
    assert hasattr(DataType, "VECTOR_UINT8"), "zvec wheel missing raw uint8 Flat storage"; \
    assert hasattr(DataType, "VECTOR_FP16"), "zvec wheel missing FP16 Flat storage"; \
    p = zvec.VamanaIndexParam(flat_data_type=DataType.VECTOR_FP16, build_prefetch_offset=8, build_prefetch_lines=0, use_bulk_build=False); \
    assert p.flat_data_type == DataType.VECTOR_FP16 and p.build_prefetch_offset == 8 and p.build_prefetch_lines == 0, "zvec wheel missing FP16 refine/build-prefetch surface"; \
    assert p.two_pass_build is True, "zvec wheel does not default Vamana to two-pass build"; \
    q = zvec.VamanaQueryParam(ef_search=96); \
    assert q.prefetch_offset is None and q.prefetch_lines is None, "zvec wheel does not expose automatic query prefetch defaults"; \
    print("zvec", zvec.__version__, "uniform_uint8 c3e5945 build with two-pass/auto-prefetch defaults, FP16 refine, bypass, UniformUint4, and raw uint8 Flat APIs OK")'
