export PATH="/usr/local/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:${LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=7 python test.py --config config/rfs_phase1_scannet.yaml
