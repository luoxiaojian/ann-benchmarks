FROM ann-benchmarks

RUN apt update
RUN apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev
RUN pip3 install wheel pybind11 faiss-cpu -i https://mirrors.aliyun.com/pypi/simple/

WORKDIR /home/app
RUN git clone https://github.com/Henry-yan/kgn.git
RUN cd kgn && git checkout 199a2d2ba69d1d6f9dbd750778b9e41f5be2163a && pip3 install pykgn-1.0.0-cp310-cp310-linux_x86_64.whl