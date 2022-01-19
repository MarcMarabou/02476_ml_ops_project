FROM pytorch/torchserve:0.5.2-cpu

ARG SCRIPT_DIR=models/
ARG SCRIPT_FILE
COPY ${SCRIPT_DIR}/${SCRIPT_FILE} index_to_name.json /home/model-server/

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
USER model-server

RUN torch-model-archiver \
  --model-name=ViT \
  --version=1.0 \
  --serialized-file=/home/model-server/${SCRIPT_FILE} \
  --handler=image_classifier \
  --export-path=/home/model-server/model-store \
  --extra-files=/home/model-server/index_to_name.json

CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "ViT=ViT.mar"]
