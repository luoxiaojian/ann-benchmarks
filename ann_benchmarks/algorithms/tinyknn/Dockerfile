FROM ann-benchmarks

RUN git clone https://github.com/thomasahle/tinyknn.git
RUN sed -i 's/) nogil:/) noexcept nogil:/g' tinyknn/tinyknn/_fast_pq*.pyx
RUN pip3 install "numpy<2" && echo "numpy<2" > /tmp/tinyknn-constraints.txt
RUN cd tinyknn && PIP_CONSTRAINT=/tmp/tinyknn-constraints.txt python3 -m pip install .
RUN python3 -c 'import tinyknn'
