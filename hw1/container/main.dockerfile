FROM continuumio/miniconda3

RUN conda install pytorch torchaudio -c pytorch
RUN conda install -c huggingface -c conda-forge datasets &&\
    conda install -c conda-forge pysoundfile &&\
    conda install -c conda-forge editdistance &&\
    conda install  -c conda-forge &&\
    conda install -c conda-forge librosa==0.8.1 &&\
    conda install -c conda-forge notebook &&\
    pip install wandb &&\
    echo 'alias jn="jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root --no-browser"' >> ~/.bashrc