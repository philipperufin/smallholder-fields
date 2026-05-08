# smallholder-fields

## Overview

This repository provides the code used to produce the MozFields 2023 dataset described in [Rufin et al. 2026](doi.org/10.1088/1748-9326/ae5cb4). 

The repository is currently in the final stages of perparation and will be released as soon as possible. The published code includes scripts for

- *finetune.ipynb*: FracTAL ResUNet model fine-tuning using model checkpoint by [Wang et al. 2022](https://doi.org/10.5281/zenodo.7315089)
- *inference.ipynb*: Field delineation inference using Mozambique model checkpoint by [Rufin et al. 2026](https://doi.org/10.5281/zenodo.17531365)
- *postprocessing.ipynb*: Post-processing routines including segmentation, polygonization, and merging

The following additional resources are available: 

- [GitHub repository containing code for producing pseudo-labels](https://github.com/philipperufin/pseudo-fields)
- [FracTAL ResUNet Model checkpoint](https://doi.org/10.5281/zenodo.17531365)
- [National scale cropland fraction and field size metrics at 0.05° resolution](https://doi.org/10.5281/zenodo.18938382)
- [National scale field delineations (restricted to non-commercial research applications)](https://doi.org/10.5281/zenodo.19481408)


## Installation

The current workflow relies no Apache MXNet, which has been retired: https://mxnet.apache.org/versions/1.9.1/ and thus running the repository will require the installation of outdated module versions, and we want to highlight that this may pose risk. For creation of the environment to run the code, please use the existing environment.yml during installation:

```mamba env create -f environment.yml python=3.10```

When completed, please activate the environment and set the CUDA library paths through the LD_LIBRARY_PATH and CUDNN_PATH variable, which  describe where to find the library files for cuda toolkit and the cuda Deep Neural Network (cuDNN). Use these commands to create the env_vars.sh files:

````
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export _LD_LIBRARY_PATH=$LD_LIBRARY_PATH'  >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'unset CUDNN_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$_LD_LIBRARY_PATH'  >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset _LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
````

Alternative instructions to run MXNet on modern GPUs via Docker containers are available:
https://www.nvidia.com/en-sg/data-center/gpu-accelerated-applications/mxnet/
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/mxnet?version=24.06-py3
