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
import pytorch_lightning as pl
import skimage

# --------------------------------------------------------------------------------------------

# Functionality Import | Fundamentals
from pathlib import Path
from math import *
from PIL import Image
from datetime import datetime
from skimage.transform import resize

# Functionality Import | Torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, dataset
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
torch.set_float32_matmul_precision('high')
from keras.optimizers import Adam

# --------------------------------------------------------------------------------------------

# Functionality Import | Custom
sys.path.append('/nas-ctm01/homes/pfsousa/seg_piu')
from residualUnet import *
from metrics import dice_coef, loss_dice
#from preprocessingFunctions import *
from segmentation_functions import *

# ============================================================================================

# UNet Initialisation
resUnet = residualUNet()
opt = Adam(learning_rate = 1e-4)
resUnet.compile(optimizer = opt, loss = loss_dice, metrics=[dice_coef])
ckpt_path = "/nas-ctm01/homes/pfsousa/seg_piu/bestmodel.epoch05.hdf5"
resUnet.load_weights(ckpt_path)

#input_fp = "/nas-ctm01/datasets/public/Lung_Nodule/train/LNDb/ct"
input_fp = "/nas-ctm01/datasets/public/Lung_Nodule/test/LNDb/ct"
#output_fp = "/nas-ctm01/datasets/public/Lung_Nodule/train/LNDb/lungmask"
output_fp = "/nas-ctm01/datasets/public/Lung_Nodule/test/LNDb/lungmask"

input_list = os.listdir(input_fp)
for fp in input_list:
    data = np.load(f"{input_fp}/{fp}")
    data_pro = resize(  data, (data.shape[0], 512, 512),
                        preserve_range = True, order = 1,
                        mode = 'edge', anti_aliasing = True).astype(np.uint16)
    print(f"Processing File | {fp} | Original Shape {data.shape} | Resized Shape {data_pro.shape}")
    seg = torch.FloatTensor(lung_segmentation_scan(resUnet, data, (512, 512)))
    np.save(f"{output_fp}/{fp}", seg.numpy())