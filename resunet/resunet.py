# ============================================================================================

# Package Import
import sys
import os
import math
import copy
import argparse
import imageio
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb

# --------------------------------------------------------------------------------------------

# Functionality Import | Fundamentals
from torch import nn, einsum
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler

from functools import partial
from torch.utils import data
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from einops import rearrange
#from einops_exts import check_shape, rearrange_many
#from pytorch_lightning.loggers import TensorBoardLogger
#from rotary_embedding_torch import RotaryEmbedding

# --------------------------------------------------------------------------------------------

# Functionality Import | Custom Dataset
sys.path.append('/nas-ctm01/homes/pfsousa/data')
from data_parser import data_parser
from __init__ import get_ds
sys.path.append('/nas-ctm01/homes/pfsousa/fedseg')
from run_parser import run_parser

# ============================================================================================

# ============= HELPER FUNCTION =============

class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2 with optional residual connection"""
    def __init__(
        self,
        in_channels,
        out_channels,
        stride = 1,
        residual = False
    ):
        
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d( in_channels,
                                out_channels,
                                kernel_size = 3,
                                padding = 'same',
                                stride = stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d( out_channels,
                                out_channels,
                                kernel_size = 3,
                                padding = 'same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # For residual path when stride > 1 or in/out channels mismatch
        self.shortcut = nn.Sequential()
        if residual and (stride > 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(  in_channels,
                            out_channels,
                            kernel_size = 1,
                            stride = stride,
                            padding = 'same'),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.residual:
            out = out + self.shortcut(identity)
        
        out = F.relu(out)
        return out


class ResidualBlock(nn.Module):
    """A single residual block with 2 conv layers and identity shortcut"""
    def __init__(
        self,
        channels,
        stride = 1
    ):
        
        super().__init__()
        self.conv1 = nn.Conv2d( channels, channels,
                                kernel_size = 3,
                                padding = 'same',
                                stride = stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d( channels, channels,
                                kernel_size = 3,
                                padding = 'same')
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(  channels, channels,
                            kernel_size = 1,
                            stride = stride,
                            padding = 'same'),
                nn.BatchNorm2d(channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x) if len(self.shortcut) > 0 else x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = F.relu(out)
        return out
    
# ============================================================================================

# ============= LOSS FUNCTIONS =============

def dice_coef(y_true, y_pred, smooth=1.0):
    """Dice coefficient for PyTorch tensors"""
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1.0):
    """Dice loss = 1 - Dice coefficient"""
    return 1.0 - dice_coef(y_true, y_pred, smooth)


def combined_loss(y_true, y_pred, smooth=1.0, bce_weight=0.5):
    """Optional: Combine BCE with Dice loss (often works better for segmentation)"""
    bce = F.binary_cross_entropy(y_pred, y_true)
    dice = dice_loss(y_true, y_pred, smooth)
    return bce_weight * bce + (1 - bce_weight) * dice

# ============================================================================================

class ResidualUNet(nn.Module):
    """ResNet34-inspired UNet for 128x128 grayscale CT segmentation"""
    
    def __init__(
        self,
        data_args = None,
        run_args = None
    ):
        super().__init__()
        self.data_args = data_args
        self.run_args = run_args
        
        # --------------------------------------------------------------------------------------------

        # Encoder (Downsampling Path) Initialization
        
        # Stage 0: Initial conv + pooling
        # kernel_size=7, stride=2, same padding = (7-1)//2 = 3
        self.stage0_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.stage0_bn = nn.BatchNorm2d(64)
        self.stage0_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 1: 3 residual blocks, 64 channels (no downsampling)
        self.stage1_block1 = ResidualBlock(64, stride=1)
        self.stage1_block2 = ResidualBlock(64, stride=1)
        self.stage1_block3 = ResidualBlock(64, stride=1)
        
        # Stage 2: 4 residual blocks, 128 channels (stride=2 on first block)
        # kernel_size=3, stride=2, same padding = (3-1)//2 = 1
        self.stage2_down = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage2_bn_down = nn.BatchNorm2d(128)
        self.stage2_shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)  # kernel=1 no padding needed
        self.stage2_block1 = ResidualBlock(128, stride=1)
        self.stage2_block2 = ResidualBlock(128, stride=1)
        self.stage2_block3 = ResidualBlock(128, stride=1)
        self.stage2_block4 = ResidualBlock(128, stride=1)
        
        # Stage 3: 6 residual blocks, 256 channels
        self.stage3_down = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.stage3_bn_down = nn.BatchNorm2d(256)
        self.stage3_shortcut = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        self.stage3_block1 = ResidualBlock(256, stride=1)
        self.stage3_block2 = ResidualBlock(256, stride=1)
        self.stage3_block3 = ResidualBlock(256, stride=1)
        self.stage3_block4 = ResidualBlock(256, stride=1)
        self.stage3_block5 = ResidualBlock(256, stride=1)
        self.stage3_block6 = ResidualBlock(256, stride=1)
        
        # Stage 4: 3 residual blocks, 512 channels (bottleneck)
        self.stage4_down = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.stage4_bn_down = nn.BatchNorm2d(512)
        self.stage4_shortcut = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)
        self.stage4_block1 = ResidualBlock(512, stride=1)
        self.stage4_block2 = ResidualBlock(512, stride=1)
        self.stage4_block3 = ResidualBlock(512, stride=1)

        # --------------------------------------------------------------------------------------------

        # Decoder (Upsampling Path) Initialization
                
        # Up 6: 512 -> 256
        self.up6_transpose = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        # padding='same' for kernel=3 => padding=1
        self.up6_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up6_bn1 = nn.BatchNorm2d(256)
        self.up6_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.up6_bn2 = nn.BatchNorm2d(256)
        
        # Up 7: 256 -> 128
        self.up7_transpose = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.up7_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up7_bn1 = nn.BatchNorm2d(128)
        self.up7_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.up7_bn2 = nn.BatchNorm2d(128)
        
        # Up 8: 128 -> 64
        self.up8_transpose = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.up8_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up8_bn1 = nn.BatchNorm2d(64)
        self.up8_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.up8_bn2 = nn.BatchNorm2d(64)
        
        # Up 9: 64 -> 64 (with skip from stage0_bn)
        self.up9_transpose = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)
        # kernel_size=7, padding='same' => padding=3
        self.up9_conv1 = nn.Conv2d(128, 64, kernel_size=7, padding=3)
        self.up9_bn1 = nn.BatchNorm2d(64)
        self.up9_conv2 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
        self.up9_bn2 = nn.BatchNorm2d(64)
        
        # Up 10: 64 -> 32 -> 16
        self.up10_transpose = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.up10_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.up10_bn1 = nn.BatchNorm2d(16)
        self.up10_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.up10_bn2 = nn.BatchNorm2d(16)
        
        # Final output
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        if run_args.verbose: self.summary()
    
    # ============================================================================================

    def forward(self, x):
        
        # Encoder Forward Pass
        
        # Stage 0
        s0_conv = self.stage0_conv(x)
        s0_bn = self.stage0_bn(s0_conv)
        s0_relu = F.relu(s0_bn)
        s0_pool = self.stage0_pool(s0_relu)
        
        # Stage 1 (skip: after all 3 blocks)
        s1 = self.stage1_block1(s0_pool)
        s1 = self.stage1_block2(s1)
        s1_out = self.stage1_block3(s1)
        
        # Stage 2 (skip: after all 4 blocks)
        s2_down = self.stage2_down(s1_out)
        s2_bn = self.stage2_bn_down(s2_down)
        s2_relu = F.relu(s2_bn)
        s2_shortcut = self.stage2_shortcut(s1_out)
        s2 = s2_relu + s2_shortcut
        s2 = F.relu(s2)
        
        s2 = self.stage2_block1(s2)
        s2 = self.stage2_block2(s2)
        s2 = self.stage2_block3(s2)
        s2_out = self.stage2_block4(s2)
        
        # Stage 3 (skip: after all 6 blocks)
        s3_down = self.stage3_down(s2_out)
        s3_bn = self.stage3_bn_down(s3_down)
        s3_relu = F.relu(s3_bn)
        s3_shortcut = self.stage3_shortcut(s2_out)
        s3 = s3_relu + s3_shortcut
        s3 = F.relu(s3)
        
        s3 = self.stage3_block1(s3)
        s3 = self.stage3_block2(s3)
        s3 = self.stage3_block3(s3)
        s3 = self.stage3_block4(s3)
        s3 = self.stage3_block5(s3)
        s3_out = self.stage3_block6(s3)
        
        # Stage 4 (bottleneck)
        s4_down = self.stage4_down(s3_out)
        s4_bn = self.stage4_bn_down(s4_down)
        s4_relu = F.relu(s4_bn)
        s4_shortcut = self.stage4_shortcut(s3_out)
        s4 = s4_relu + s4_shortcut
        s4 = F.relu(s4)
        
        s4 = self.stage4_block1(s4)
        s4 = self.stage4_block2(s4)
        s4_out = self.stage4_block3(s4)

        # --------------------------------------------------------------------------------------------
        
        # Decoder Forward Pass
        
        # Up 6: connect with s3_out
        d6 = self.up6_transpose(s4_out)
        d6 = torch.cat([d6, s3_out], dim=1)
        d6 = self.up6_conv1(d6)
        d6 = self.up6_bn1(d6)
        d6 = F.relu(d6)
        d6 = self.up6_conv2(d6)
        d6 = self.up6_bn2(d6)
        d6 = F.relu(d6)
        
        # Up 7: connect with s2_out
        d7 = self.up7_transpose(d6)
        d7 = torch.cat([d7, s2_out], dim=1)
        d7 = self.up7_conv1(d7)
        d7 = self.up7_bn1(d7)
        d7 = F.relu(d7)
        d7 = self.up7_conv2(d7)
        d7 = self.up7_bn2(d7)
        d7 = F.relu(d7)
        
        # Up 8: connect with s1_out
        d8 = self.up8_transpose(d7)
        d8 = torch.cat([d8, s1_out], dim=1)
        d8 = self.up8_conv1(d8)
        d8 = self.up8_bn1(d8)
        d8 = F.relu(d8)
        d8 = self.up8_conv2(d8)
        d8 = self.up8_bn2(d8)
        d8 = F.relu(d8)
        
        # Up 9: connect with s0_relu
        d9 = self.up9_transpose(d8)
        d9 = torch.cat([d9, s0_relu], dim=1)
        d9 = self.up9_conv1(d9)
        d9 = self.up9_bn1(d9)
        d9 = F.relu(d9)
        d9 = self.up9_conv2(d9)
        d9 = self.up9_bn2(d9)
        d9 = F.relu(d9)
        
        # Up 10: final upsampling to original resolution
        d10 = self.up10_transpose(d9)
        d10 = self.up10_conv1(d10)
        d10 = self.up10_bn1(d10)
        d10 = F.relu(d10)
        d10 = self.up10_conv2(d10)
        d10 = self.up10_bn2(d10)
        d10 = F.relu(d10)
        
        # Output
        out = self.final_conv(d10)
        out = torch.sigmoid(out)
        
        return out
    
    # ============================================================================================
    
    # Model Summary Function
    def summary(self):

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model: ResidualUNet")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Print layer shapes (simplified summary)
        print("\nEncoder stages:")
        print("  Stage 0: Conv7x7(1->64) + MaxPool -> 32x32x64")
        print("  Stage 1: 3x ResidualBlock(64) -> 32x32x64")
        print("  Stage 2: Downsample + 4x ResidualBlock(128) -> 16x16x128")
        print("  Stage 3: Downsample + 6x ResidualBlock(256) -> 8x8x256")
        print("  Stage 4: Downsample + 3x ResidualBlock(512) -> 4x4x512")
        print("\nDecoder stages:")
        print("  Up 6: 512->256 + skip(256) -> 8x8x256")
        print("  Up 7: 256->128 + skip(128) -> 16x16x128")
        print("  Up 8: 128->64 + skip(64) -> 32x32x64")
        print("  Up 9: 64->64 + skip(64) -> 64x64x64")
        print("  Up 10: 64->32->16 -> 128x128x16")
        print("  Output: Conv1x1(16->1) + Sigmoid -> 128x128x1")