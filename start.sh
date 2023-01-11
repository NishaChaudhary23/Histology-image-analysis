#! /bin/bash
#PBS -N histology21
#PBS -o model21_out.log
#PBS -e model21_err.log
#PBS -l ncpus=10
#PBS -q gpu

module load compiler/anaconda3

conda init

source ~/.bashrc

conda activate histoimgan

python3 /storage/bic/data/oscc/data/Histology-image-analysis/model.py 