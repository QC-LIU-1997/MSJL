# MSJL
This repository provides PyTorch implementation for the paper entitled "MSJL for paper entitled "Multisource Space-frequency Joint Learning: A Novel Paradigm for Ultrasound Image Quality Assessment", which is under review. 

## MSJL_pretraining.py
How to pretrain backbone with soft masked frequency modeling (SMFM), taking ResNet as example.

## MSJL_finetuning.py
On-the-fly joint finetuning with FAF (frequency-aware fusion).

## Tips:
The code can run on linux or windows systems. We recommend torch == 1.12.1 and timm == 0.6.13.

## Acknowledgement
We thank research from [MFM](https://github.com/Jiahao000/MFM) and [FcaNet](https://github.com/cfzd/FcaNet), whose code also inspired our work.
