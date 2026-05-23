#!/bin/bash
#
#SBATCH --partition=gpu_min8gb
#SBATCH --output=/nas-ctm01/homes/pfsousa/fedseg/logs/run_VI/output.out
#SBATCH --error=/nas-ctm01/homes/pfsousa/fedseg/logs/run_VI/error.err
#SBATCH --job-name=fedseg
#SBATCH --time=1-00:00
#SBATCH --qos=gpu_min8GB

conda init --all
conda activate meddiff3

/nas-ctm01/homes/pfsousa/.conda/envs/meddiff3/bin/python /nas-ctm01/homes/pfsousa/fedseg/runs/main_VI.py
#python main_VI.py