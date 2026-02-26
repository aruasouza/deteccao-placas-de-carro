FROM ubuntu:24.04

RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update && \
    apt install -y python3.12 python3.12-venv python3.12-dev python3-pip python3-opencv git wget

WORKDIR /home/ubuntu

RUN python3.12 -m venv /home/ubuntu/venv && \
    /home/ubuntu/venv/bin/pip install --upgrade pip setuptools wheel

COPY ./rknn_toolkit_wheels/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl .

RUN /home/ubuntu/venv/bin/pip install rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl && \
    rm rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl && \
    /home/ubuntu/venv/bin/pip install setuptools==80.9.0 onnx==1.18.0 onnxruntime==1.18.0

WORKDIR /home/ubuntu

COPY run.sh /home/ubuntu/run.sh

RUN chmod +x /home/ubuntu/run.sh

CMD ["/home/ubuntu/run.sh"]
