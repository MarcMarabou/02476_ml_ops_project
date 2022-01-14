#!/bin/bash
export REGION=europe-north1
export JOB_NAME=vit_model_$(date +%Y%m%d_%H%M%S)
export IMAGE_URI=gcr.io/aerobic-datum-337911/vit_trainer:latest
gcloud ai-platform jobs submit training $JOB_NAME \
    --region $REGION \
    --master-image-uri $IMAGE_URI \
    -- \
    --depth=1 \
    --num-heads=1 \
    --embed-dim=1 \
    --num-workers=8 \
    --data-path="gs://dtu-ml-ops-2022-10/processed/processed/flowers/" \
    --model-dir="gs://dtu-ml-ops-2022-10/models/"