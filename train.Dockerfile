## Conda setup
#FROM continuumio/miniconda3:latest
#RUN gcloud auth activate-service-account g27-bucket@mlops-g27.iam.gserviceaccount.com --key-file=key_file.json
#FROM gcr.io/cloud-builders/gsutil

#RUN dvc remote modify --local remote_storage \
#        credentialpath key_file.json

# GPU runtime
FROM nvidia/cuda:10.2-cudnn8-runtime

# Install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc wget && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

COPY requirements_docker.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/

ENV PATH=/root/miniconda3/bin:$PATH
ARG PATH=/root/miniconda3/bin:$PATH

ENV GOOGLE_APPLICATION_CREDENTIALS keys/aerobic-datum-337911-fcd8e3b6bec8.json
ENV GOOGLE_PROJECT_ID aerobic-datum-337911-fcd8e3b6bec8

RUN wget -nv \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda install python==3.9.7
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
RUN echo '[Credentials]\ngs_service_key_file = /keys/aerobic-datum-337911-fcd8e3b6bec8.json' \ 
    > /etc/boto.cfg
RUN mkdir /keys

COPY keys/aerobic-datum-337911-fcd8e3b6bec8.json $GOOGLE_APPLICATION_CREDENTIALS

RUN gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS --project $GOOGLE_PROJECT_ID
RUN gcloud config set project $GOOGLE_PROJECT_ID

ENTRYPOINT [ "python", "-u", "src/models/train_model.py" ]