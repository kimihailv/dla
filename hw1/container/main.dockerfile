FROM pytorch/pytorch

RUN pip install torchaudio editdistance librosa wandb youtokentome gdown
RUN git clone --recursive https://github.com/parlance/ctcdecode.git && cd ctcdecode && pip install .
RUN git https://github.com/kimihailv/dla.git
RUN echo 'alias jn="jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root --no-browser"' >> ~/.bashrc
