FROM ann-benchmarks

RUN apt update
RUN apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev
RUN pip3 install wheel pybind11==2.5.0

WORKDIR /home/app
COPY git_deps/descartes/descartes /home/app/descartes
RUN pip3 install descartes/descartes-*-linux_x86_64.whl
