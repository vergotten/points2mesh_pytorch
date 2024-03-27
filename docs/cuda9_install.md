#### CUDA and cuDNN

The script installs CUDA 9.0. For cuDNN, it extracts the cuDNN files from a specified path and copies them to the CUDA directory.

**Important:** Please specify the path to your cuDNN files in the script. The current path is `/tools/cudnn-9.0-linux-x64-v7.3.0.29.tgz`. If your cuDNN files are located elsewhere, please update this path in the script.

#### Usage

To use the script, first make it executable with the following command:

```bash
chmod +x scripts/setup_ubuntu1604.sh
```

Then, you can run the script with:

```bash
./scripts/setup_ubuntu1604.sh
```

## Anaconda Environment Setup

Details about the `setup_conda_env.sh` script, which installs Anaconda, creates a new environment `myenv`, installs requirements, and builds libraries `lib` from CUDA for efficient computing.

## Testing

Instructions on how to test if CUDA is working as expected in the container:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

```bash
# Check if the libraries are installed and working
python -c "
try:
    import torch
    import torchvision
    import spconv
    from lib.pointgroup_ops.functions import pointgroup_ops
    import bspt
    print('All libraries are installed and working.')
except ImportError as e:
    print(f'Error: {e}')
"
```