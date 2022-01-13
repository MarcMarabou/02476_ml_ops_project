#!/bin/bash
export REGION=europe-north1
export JOB_NAME=vit_model_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME \
    --region $REGION \
    --master-image-uri $IMAGE_URI \
    -- \
    --depth=1 \
    --num-heads=1 \
    --embed-dim=1 \
    --num-workers=8 \
    --model-dir="gs://dtu-ml-ops-2022-10/models/"