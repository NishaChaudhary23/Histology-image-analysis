#! /bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chs.rintu/anaconda3/envs/histoimgan/lib
python3 cluster.py
python3 plot.py