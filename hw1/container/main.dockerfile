FROM pytorch/pytorch

RUN apt-get update && apt-get install -y build-essential && apt-get install -y git
RUN pip install editdistance librosa wandb youtokentome gdown
RUN git clone --recursive https://github.com/parlance/ctcdecode.git && cd ctcdecode && pip install .
RUN git clone https://github.com/kimihailv/dla.git
RUN echo 'alias jn="jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root --no-browser"' >> ~/.bashrc
