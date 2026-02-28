#!/bin/bash

# sift
python run.py --algorithm qsgngt --dataset sift-128-euclidean
# python run.py --algorithm parlayann --dataset sift-128-euclidean
python run.py --algorithm kgn --dataset sift-128-euclidean
python run.py --algorithm 'descartes(01AI)' --dataset sift-128-euclidean
# python run.py --algorithm glass --dataset sift-128-euclidean

python plot.py --dataset sift-128-euclidean \
    --output results/sift.png --x-scale logit --y-scale log
