FROM almalinux:8

ARG VESPA_ANN_BENCHMARK_VERSION=8.736.12-1.el8

RUN dnf -y install epel-release && \
    dnf -y install dnf-plugins-core && \
    dnf -y install \
        gcc \
        make \
        git \
        python3.11-devel \
        python3.11-pip \
        python3.11-setuptools && \
    dnf config-manager --add-repo https://raw.githubusercontent.com/vespa-engine/vespa/master/dist/vespa-engine.repo && \
    dnf -y install --enablerepo=powertools \
        "vespa-ann-benchmark-${VESPA_ANN_BENCHMARK_VERSION}"

WORKDIR /home/app

COPY requirements.txt run_algorithm.py ./

RUN python3.11 -m pip install -r requirements.txt && \
    python3.11 -m pip install /opt/vespa/libexec/vespa_ann_benchmark

# vespa_ann_benchmark links libvespalib with initial-exec TLS.  Load it before
# CPython imports other extension DSOs so glibc reserves its static TLS block.
# This changes only process startup; the Vespa package and benchmark path stay
# identical to the upstream adapter.
ENV LD_PRELOAD=/opt/vespa/lib64/libvespalib.so

ENTRYPOINT ["python3.11", "-u", "run_algorithm.py"]
