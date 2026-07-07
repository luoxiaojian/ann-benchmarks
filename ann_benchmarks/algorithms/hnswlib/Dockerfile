FROM ann-benchmarks

RUN pip install pybind11 numpy setuptools
COPY git_deps/hnswlib/hnsw /home/app/hnsw

RUN cd hnsw/python_bindings; python3 setup.py install

RUN python -c 'import hnswlib'

