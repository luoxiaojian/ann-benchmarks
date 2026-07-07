FROM ubuntu:22.04

# Use Chinese apt mirror
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list

RUN apt-get update
RUN apt-get install -y python3-pip python-is-python3 build-essential git curl
RUN python --version | grep 'Python 3.10'

# Configure pip to use Chinese mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "-u", "run_algorithm.py"]
