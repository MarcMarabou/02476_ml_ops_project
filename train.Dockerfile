# GPU runtime
FROM nvidia/cuda:10.2-cudnn8-runtime

## Conda setup
#FROM continuumio/miniconda3:latest

FROM python:3.9.7-slim

# Install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc wget && \
apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

COPY environment.yml environment.yml
COPY requirements_docker.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/

# Figure out why conda is so awful in a docker container
# RUN conda env update --name ml_ops --file environment.yml
# SHELL ["conda", "run", "--no-capture-output", "-n", "ml_ops", "/bin/bash", "-c"]

RUN pip install --upgrade pip --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir

# Installs google cloud sdk
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /app/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /app/tools && \
    rm google-cloud-sdk.tar.gz && \
    /app/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /app/.config/* && \
    ln -s /app/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /app/tools/google-cloud-sdk/.install/.backup

ENV PATH $PATH:/app/tools/google-cloud-sdk/bin
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ENTRYPOINT [ "python", "-u", "src/models/train_model.py" ]