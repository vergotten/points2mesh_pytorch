export PATH="/usr/local/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:${LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=5 python test.py --config config/rfs_phase2_scannet.yaml