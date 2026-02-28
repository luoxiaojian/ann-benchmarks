FROM ann-benchmarks

# Install zvec dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install zvec
RUN pip install --no-cache-dir zvec

# Verify installation
RUN python3 -c 'import zvec; print("zvec version:", zvec.__version__)'
