#! /bin/bash
#PBS -N FINAL_tes
#PBS -l host=compute4
#PBS -o /storage/bic/data/oscc/data/Histology-image-analysis/training/model_pool_dense_dense_1.log
#PBS -e /storage/bic/data/oscc/data/Histology-image-analysis/training/model_pool_dense_dense_1.log
#PBS -q gpu

module load compiler/anaconda3

conda init

source ~/.bashrc

conda activate histoimgan
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rintu.kutum/.conda/envs/histoimgan/lib/python3.10/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rintu.kutum/.conda/envs/histoimgan/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rintu.kutum/.conda/envs/histoimgan/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rintu.kutum/.conda/pkgs/cudatoolkit-11.2.2-hbe64b41_11/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rintu.kutum/.conda/pkgs/cudatoolkit-11.0.3-h88f8997_11/lib
python3 /storage/bic/data/oscc/data/Histology-image-analysis/training/model_1.py 