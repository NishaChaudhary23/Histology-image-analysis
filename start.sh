#! /bin/bash
#PBS -N histology1
#PBS -o model1_out.log
#PBS -e model1err.log
#PBS -l ncpus=10
#PBS -q gpu

module load compiler/anaconda3

conda init

source ~/.bashrc

conda activate histoimgan

python3 /storage/bic/data/oscc/data/Histology-image-analysis/model.py 