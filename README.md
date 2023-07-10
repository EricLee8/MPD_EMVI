# Codes and data for ACL 2023
This repository contains the official codes for our paper at ACL 2023: Pre-training Multi-party Dialogue Models with Latent Discourse Inference.

## Overview
This repository contains the codes and data for pre-training in the `./pre-training` folder, and the codes and data for downstream tasks in the `./downstream` folder. You can find instructions to run the code in the corresponding folder.

## Environment
Our experiments are conducted in 8 NVIDIA A100 40GB GPUs. The GPU memory consumption for single card is around 30GB. If you do not have enough GPUs or memory, you can reduce the batch size and modify the `accelerator_config.yml` to the GPU number you have. In this process, you'd better reduce the learning rate proportionally.

The python, pytorch, and CUDA version are as follows:
```
python: 3.8.12
torch: 1.10.0+cu113
CUDA: 11.3
```
Other dependencies can be found in `requirements.txt`.

We recommend that you set up the environment using anaconda. You can run the following commands:
```
conda create -n emvi python=3.8.12
conda activate emvi
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch/
pip install -r requirements.txt
```

## Usage
After setting up the environment, you can move into the corresponding folders to read further instructions and run the code.

## Citation
If you find our paper and repository useful, please cite us in your paper:
```
@inproceedings{li-etal-2023-pre,
    title = "Pre-training Multi-party Dialogue Models with Latent Discourse Inference",
    author = "Li, Yiyang  and
      Huang, Xinting  and
      Bi, Wei  and
      Zhao, Hai",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.533",
    pages = "9584--9599",
}
```
