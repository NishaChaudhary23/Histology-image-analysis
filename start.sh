#! /bin/bash
#PBS -N histology
#PBS -o model_out.log
#PBS -e model_err.log
#PBS -l ncpus=100
#PBS -q cpu

module load compiler/anaconda3

conda init

source ~/.bashrc

conda activate histoimgan

python3 /storage/bic/data/oscc/data/Histology-image-analysis/model.py 