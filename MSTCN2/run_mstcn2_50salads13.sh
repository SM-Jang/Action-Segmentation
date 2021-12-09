#!/bin/bash
dataset='50salads'
split=4
method='triplet_13'
op='residual'
gpu=3


python main.py --action=train --dataset=$dataset --split=$split --method=$method --op=$op --gpu=$gpu\
                --num_epochs=50 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3

python main.py --action=predict --dataset=$dataset --split=$split --method=$method --op=$op --gpu=$gpu\
                --num_epochs=50 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3
                
python eval.py --dataset=$dataset --split=$split --method=$method --op=$op