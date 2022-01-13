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

##Copy and install dependencies to downlaod dataset
COPY req_downlaod.txt req_downlaod.txt
RUN pip install -r req_downlaod.txt --no-cache-dir
RUN pip install dvc[gs]

## download dataset raw into docker image
## that wa we dont have to download it every time we pull the data
## downside big docker image...
RUN mkdir -p /app/src/data/raw && mkdir -p /app/data/preprocessed
COPY .dvc /app/.dvc
COPY data/preprocessed.dvc /app/data/preprocessed.dvc
# RUN dvc config core.no_scm true
# RUN dvc pull


##Copy and install model dependencies
## tat way the image until here stays the same and does not need to be rebuild if we change the architecture
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY src/ /app/src/

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]