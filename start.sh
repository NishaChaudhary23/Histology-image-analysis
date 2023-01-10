#! /bin/bash
#PBS -N HistoImageAnalysis
#PBS -o out.log
#PBS -e err.log
#PBS -l ncpus=1
#PBS -q gpu

module load compiler/anaconda3
source histoimgan/bin/activate

python3 /storage/bic/data/oscc/data/Histology-image-analysis/model.py train_model "DenseNet121"