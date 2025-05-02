# FreePoint-mmdet3d
Unofficial implementation of FreePoint: Unsupervised Point Cloud Instance Segmentation by Zhang et al. built on `mmdet3d`.

## Setup

### Dependencies


Python 3.10
```bash
python3 -m venv ./env
source ./env/bin/activate
```

```bash
pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install mmengine==0.9.0
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmdet==3.2.0
pip install mmdet3d==1.4.0
pip install -r requirements.txt
pip install numpy==1.23.5
pip install numba==0.60.0
```

_pytorch3d_
```bash
pip install iopath fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html

```


_Geometric Helpers_
```bash
pip install torch-geometric==2.5.3
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```



_RAMA installation_
> Set ENV vars
```bash
export CUDA_HOME=/usr/local/cuda-11.8/
export CUDA_PATH=${CUDA_HOME}
export CUDAToolkit_ROOT=${CUDA_HOME}
export PATH=${CUDA_HOME}bin${PATH:+:${PATH}}
export CPATH=${CUDA_HOME}include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

> Install from git

`pip install git+https://github.com/pawelswoboda/RAMA.git@2fa8b71d301ac8e7382a089a9eb8350755851a42`



## Quickstart

```bash
>> from free_point.utils.run import *
>> run(
    out_dir="output/"
    weight_feature=0,
    weight_xyz=1,
    k1=4,
    k2=4,
    num_points=20000,
    num_iters_for_graph_cut=5,
    ransac_distance=0.175
)

```
