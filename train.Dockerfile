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

## download dataset raw into docker image
## that wa we dont have to download it every time we pull the data
## downside big docker image...
RUN mkdir -p src/data && mkdir -p data/raw
COPY src/data /app/src/data
COPY data/ /app/data/
RUN python src/data/make_dataset.py

##Copy and install model dependencies
## tat way the image until here stays the same and does not need to be rebuild if we change the architecture
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY src/ src/

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]