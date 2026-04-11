FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Basic tools
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    git wget curl vim net-tools \
    gcc g++ make cmake \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# CANN toolkit will be mounted or installed separately since it's large
# Set CANN environment variables (adjust version as needed)
ENV ASCEND_HOME=/usr/local/Ascend
ENV PATH=${ASCEND_HOME}/ascend-toolkit/latest/bin:${PATH}
ENV LD_LIBRARY_PATH=${ASCEND_HOME}/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH}

# Python packages for our simulator + profiling
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torch_npu \
    transformers \
    networkx \
    pytest \
    pandas \
    numpy

# Create workspace
WORKDIR /workspace
RUN mkdir -p /workspace/xpu-simulator /workspace/profiling_data

# Copy our simulator code (will be done at build time or via mount)
COPY . /workspace/xpu-simulator/

EXPOSE 8080

CMD ["/bin/bash"]
