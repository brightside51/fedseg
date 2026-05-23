#!/bin/bash
#
#SBATCH --partition=gpu_min8gb
#SBATCH --output=/nas-ctm01/homes/pfsousa/seg_piu/output.out
#SBATCH --error=/nas-ctm01/homes/pfsousa/seg_piu/error.err
#SBATCH --job-name=seg_piu
#SBATCH --time=1-00:00
#SBATCH --qos=gpu_min8GB

python main_VI.py