FROM python:3.7-slim

# install python
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

##Copy and install cookie cutter dependencies
COPY requirements_local.txt requirements_local.txt
COPY setup.py setup.py
RUN pip install -r requirements_local.txt --no-cache-dir

## download dataset raw into docker image
## that wa we dont have to download it every time we pull the data
## downside big docker image...
RUN mkdir -p /app/src/data/raw && mkdir -p /app/data/preprocessed
COPY .dvc /app/.dvc
COPY data/preprocessed.dvc /app/data/preprocessed.dvc
RUN dvc config core.no_scm true
RUN dvc pull

##Copy and install dependencies to predict
COPY requirements_predict.txt requirements_predict.txt
RUN pip install -r requirements_predict.txt --no-cache-dir

RUN mkdir -p /app/config
COPY config /app/config

COPY src/ /app/src/
ARG WANDB_TOKEN
ENV WANDB_API=$WANDB_TOKEN

ENTRYPOINT ["python", "-u", "src/models/predict_model.py"]