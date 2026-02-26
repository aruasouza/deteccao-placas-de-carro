#!/bin/bash
if [ -d "deteccao-placas-de-carro" ]; then
    cd deteccao-placas-de-carro
    git pull
else
    git clone https://github.com/aruasouza/deteccao-placas-de-carro/
    cd deteccao-placas-de-carro
fi
/home/ubuntu/venv/bin/python onnx_to_rknn.py
