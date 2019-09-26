FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN apt-get update && \
  conda env create -f conda_env.yml
  echo "source activate swifr" >> ~/.bashrc

ENV PATH /opt/conda/envs/swifr/bin:$PATH
