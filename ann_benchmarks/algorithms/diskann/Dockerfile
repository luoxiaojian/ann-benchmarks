FROM ann-benchmarks

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y wget git cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-dev python3 python3-setuptools python3-pip libmkl-full-dev
RUN pip3 install pybind11 numpy

# The legacy DiskANN branch expects Intel's old /opt layout. Ubuntu's MKL
# package provides the same headers and libraries under /usr.
RUN mkdir -p /opt/intel/compilers_and_libraries/linux/mkl/lib /opt/intel/compilers_and_libraries/linux/lib && \
    ln -s /usr/include/mkl /opt/intel/compilers_and_libraries/linux/mkl/include && \
    ln -s /usr/lib/x86_64-linux-gnu /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64 && \
    ln -s /usr/lib/x86_64-linux-gnu /opt/intel/compilers_and_libraries/linux/lib/intel64
ENV MKL_THREADING_LAYER=GNU

RUN git clone --single-branch --branch python_bindings https://github.com/microsoft/diskann
RUN mkdir -p diskann/build
RUN cd diskann/build && cmake -DCMAKE_BUILD_TYPE=Release ..
RUN cd diskann/build && make -j
RUN cd diskann/python && pip install -e .
RUN python3 -c 'import vamanapy'
