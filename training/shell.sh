conda activate histoimgan
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chs.rintu/anaconda3/anaconda3/envs/histoimgan/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/anaconda3/pkgs/cudatoolkit-11.2.2-hbe64b41_11/lib
python model.py
# libcudart.so