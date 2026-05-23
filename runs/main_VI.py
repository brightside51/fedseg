# ============================================================================================

# Package Import
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
import torch
import wandb

# --------------------------------------------------------------------------------------------

# Functionality Import | Fundamentals
from pathlib import Path
from math import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, dataset
from datetime import datetime

# Functionality Import | Torch
#from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
torch.set_float32_matmul_precision('high')

# --------------------------------------------------------------------------------------------

# Functionality Import | Custom
sys.path.append('/nas-ctm01/homes/pfsousa/data')
from data_parser import data_parser
from __init__ import get_ds
sys.path.append('/nas-ctm01/homes/pfsousa/fedseg')
from run_parser import run_parser
sys.path.append('/nas-ctm01/homes/pfsousa/fedseg/resunet')
from resunet import ResidualUNet
sys.path.append('/nas-ctm01/homes/pfsousa/fedseg/train')
from train_resunet import train_resunet

# ============================================================================================

# Argument Initialisation
data_args = data_parser(dataset = 'covid_jun2020', dataV = 'VI', save = True)
run_args = run_parser(model = 'fedseg', runV = 'VI', save = True)
print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list())

# Dataset Initialisation
#ds = get_ds(data_args, mode = 'train')
#ct, mask = ds.__getitem__(0)
#print(ct.shape); print(mask.shape)

# Model Initialisation Example
#model = ResidualUNet(data_args, run_args)
#print(model)

# WandB Setup
wandb.login()
run_logger = wandb.init(entity = "brightside51", project = run_args.model,          # 
                        name = f"{run_args.runV} ({datetime.now().strftime('%H:%M %d/%m/%Y')})",
                        config = {"dataV": data_args.dataV, "runV": run_args.runV,})

# --------------------------------------------------------------------------------------------

# ResUNet Training Script
train_resunet(data_args, data_args, run_args, run_logger)
