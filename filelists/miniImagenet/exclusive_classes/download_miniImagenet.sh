#!/usr/bin/env bash
wget https://raw.githubusercontent.com/twitter-research/meta-learning-lstm/master/data/miniImagenet/test.csv
wget https://raw.githubusercontent.com/twitter-research/meta-learning-lstm/master/data/miniImagenet/train.csv
wget https://raw.githubusercontent.com/twitter-research/meta-learning-lstm/master/data/miniImagenet/val.csv
# wget http://image-net.org/image/ILSVRC2015/ILSVRC2015_CLS-LOC.tar.gz -P ../../Datasets/
# tar -zxvf ILSVRC2015_CLS-LOC.tar.gz -C ../../Datasets/
# python make_json.py