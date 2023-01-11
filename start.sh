#! /bin/bash
#PBS -N histology3
#PBS -o model3_out.log
#PBS -e model3_err.log
#PBS -l ncpus=10
#PBS -q gpu

module load compiler/anaconda3

conda init

source ~/.bashrc

conda activate histoimgan

python3 /storage/bic/data/oscc/data/Histology-image-analysis/model.py 