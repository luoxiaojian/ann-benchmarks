FROM ann-benchmarks
RUN pip install weaviate-client[grpc]==3.16.0

# weaviate-client 3.16.0 otherwise downloads this exact binary on first use.
# Fetch it while building, pin its digest, and place it at the cache path the
# unmodified EmbeddedOptions(version="1.19.0-beta.1") code already derives.
ARG WEAVIATE_EMBEDDED_URL=https://github.com/weaviate/weaviate/releases/download/v1.19.0-beta.1/weaviate-v1.19.0-beta.1-linux-amd64.tar.gz
ARG WEAVIATE_EMBEDDED_SHA256=d9df0edd9bbd4e4b7bbf4989c054423fbf7b98dc016ad47801b91b75ce31e3a2
COPY ann_benchmarks/algorithms/weaviate/install_embedded.py /tmp/install_weaviate_embedded.py
RUN python /tmp/install_weaviate_embedded.py \
        --version 1.19.0-beta.1 \
        --url "$WEAVIATE_EMBEDDED_URL" \
        --sha256 "$WEAVIATE_EMBEDDED_SHA256" \
        --cache-dir /root/.cache/weaviate-embedded && \
    rm /tmp/install_weaviate_embedded.py
