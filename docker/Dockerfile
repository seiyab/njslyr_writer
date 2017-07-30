FROM continuumio/miniconda3:latest

MAINTAINER seiyab

RUN apt-get update && apt-get install --yes \
    libmecab-dev \
    mecab \
    mecab-ipadic \
    mecab-ipadic-utf8 \
    python-mecab \
 && rm -rf /var/lib/apt/lists/*

# python libraries
RUN /opt/conda/bin/conda install --yes \
    ipython \
    numpy \
 && /opt/conda/bin/conda clean --yes --all
RUN pip install --no-cache-dir\
    chainer \
    tweepy

# mecab config
COPY chado_utf8.csv /mecab-dict/chado.csv
RUN /usr/lib/mecab/mecab-dict-index \
    -d /var/lib/mecab/dic/ipadic-utf8/ \
    -u /mecab-dict/chado.dic \
    -f utf8 \
    -t utf8 /mecab-dict/chado.csv \
 && echo "userdic = /mecab-dict/chado.dic" >> /var/lib/mecab/dic/ipadic-utf8/dicrc


WORKDIR /src

ENTRYPOINT /bin/bash
