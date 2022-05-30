# docker build -t pet_biometric:0.1.0 .
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

WORKDIR /home/pet_biometric/train
ADD .  /home/pet_biometric/train
RUN mkdir -p ../data && rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && cd depends && mv sources.list /etc/apt/ && mv .pip/ /root/.pip/ && apt-get update &&  \
        apt-get install -y python3 python3-pip curl libgl1-mesa-glx libopenblas-dev libomp-dev zip libsm6 libxrender1 && pip3 install -U pip Cython && cd .. && pip3 install -r requirements.txt && \
        ln -s /usr/bin/python3 /usr/bin/python && python setup.py develop  && mkdir -p /root/.cache/torch/hub/checkpoints/ && mv depends/*.pth /root/.cache/torch/hub/checkpoints/ 
