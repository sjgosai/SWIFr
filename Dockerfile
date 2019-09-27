FROM continuumio/miniconda3

WORKDIR /app

RUN git clone https://github.com/sjgosai/SWIFr.git /app

COPY ./models /app/models

RUN apt-get update && \
  conda env create -f conda_env.yml
  echo "source activate swifr" >> ~/.bashrc

ENV PATH /opt/conda/envs/swifr/bin:$PATH
