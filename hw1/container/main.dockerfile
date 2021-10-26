FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN apt-get update && apt-get install -y build-essential &&\
    apt-get install -y git && apt-get install -y libsndfile1-dev
RUN pip install editdistance librosa wandb youtokentome gdown h5py
RUN pip install torchaudio==0.9.1
RUN git clone --recursive https://github.com/parlance/ctcdecode.git && cd ctcdecode && pip install .
RUN git clone https://github.com/kimihailv/dla.git
