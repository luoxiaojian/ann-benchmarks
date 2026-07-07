FROM ann-benchmarks

RUN apt update
RUN apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev
RUN pip3 install wheel pybind11==2.5.0
COPY git_deps/qsg_ngt/HWTL_SDU-ANNS /home/app/HWTL_SDU-ANNS
RUN cp HWTL_SDU-ANNS/create.py /home/app/
RUN cp -r HWTL_SDU-ANNS/lib/* /usr/local/lib/
RUN cp -r HWTL_SDU-ANNS/bin/* /usr/local/bin/
RUN chmod a+x /usr/local/bin/* && chmod a+x HWTL_SDU-ANNS/*
RUN ldconfig
RUN pip3 install HWTL_SDU-ANNS/qsgngt-*-linux_x86_64.whl
