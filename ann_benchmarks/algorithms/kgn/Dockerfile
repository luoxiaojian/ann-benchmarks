FROM ann-benchmarks

RUN apt update
RUN apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev
RUN pip3 install wheel pybind11 faiss-cpu

WORKDIR /home/app
COPY git_deps/kgn/kgn /home/app/kgn
RUN cd kgn && git checkout 32f67d08438d76769521b47603f8c727693e532c -- pykgn-1.0.0-cp310-cp310-linux_x86_64.whl
RUN pip3 install kgn/pykgn-1.0.0-cp310-cp310-linux_x86_64.whl