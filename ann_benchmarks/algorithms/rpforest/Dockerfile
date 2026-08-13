FROM ann-benchmarks

RUN git clone https://github.com/lyst/rpforest
RUN pip3 install "Cython<3" "numpy<2" && cd rpforest && python3 setup.py cythonize
RUN cd rpforest && python3 setup.py install
RUN python3 -c 'import rpforest'
