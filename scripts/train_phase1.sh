export PATH="/usr/local/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:${LD_LIBRARY_PATH}"

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train.py --config config/rfs_phase1_scannet.yaml
