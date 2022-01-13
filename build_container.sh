#!/bin/bash

export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=vit_trainer
export IMAGE_TAG=vit_trainer_gpu
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f train.Dockerfile -t $IMAGE_URI ./