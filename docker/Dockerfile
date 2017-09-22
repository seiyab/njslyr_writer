FROM continuumio/miniconda3:latest

MAINTAINER seiyab

RUN apt-get update && apt-get install --yes \
    g++ \
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
    mecab-python3 \
    tweepy

# mecab config
COPY chado_utf8.csv /mecab-dict/chado.csv
RUN /usr/lib/mecab/mecab-dict-index \
    -d /var/lib/mecab/dic/ipadic-utf8/ \
    -u /mecab-dict/chado.dic \
    -f utf8 \
    -t utf8 /mecab-dict/chado.csv \
 && echo "userdic = /mecab-dict/chado.dic" >> /var/lib/mecab/dic/ipadic-utf8/dicrc

ENV PYTHONPATH $PYTHONPATH:/njslyr_writer

# cron
RUN apt-get update && apt-get install --yes cron \
 && rm -rf /var/lib/apt/lists/* \
 && printenv | sed 's/^\(.*\)$/export \1/g' > /root/env_vars.sh \
 && echo '00 23,1,3,5,7,9,11,13 * * * root . /root/env_vars.sh; /njslyr_writer/njslyr_writer/twitter/tweet.py >> /log 2>&1' > /etc/cron.d/twitter-crontab \
 && echo '*/3 * * * * root . /root/env_vars.sh; /njslyr_writer/njslyr_writer/twitter/reply.py >> /log 2>&1' >> /etc/cron.d/twitter-crontab \
 && chmod 644 /etc/cron.d/twitter-crontab

ENTRYPOINT ["cron", "-f"]
