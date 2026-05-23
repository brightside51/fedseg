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
import hydra
import torch.distributed as dist

# --------------------------------------------------------------------------------------------

# Functionality Import | Fundamentals
from pathlib import Path
from math import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, dataset
from datetime import datetime
from omegaconf import DictConfig, open_dict

# Functionality Import | Torch
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# --------------------------------------------------------------------------------------------

# Functionality Import | Custom
sys.path.append('/nas-ctm01/homes/pfsousa/data')
from data_parser import data_parser
from __init__ import get_ds
sys.path.append('/nas-ctm01/homes/pfsousa/fedseg')
from run_parser import run_parser
sys.path.append('/nas-ctm01/homes/pfsousa/fedseg/resunet')
from resunet import ResidualUNet, dice_coef, dice_loss, combined_loss

# ============================================================================================

def train_resunet(
    train_args = None,
    val_args = None,
    run_args = None,
    run_logger = None
):
    
    # Training Dataset Initialisation
    if type(train_args) == list:
        train_ds = []
        for train_arg in train_args:
            train_ds.append(get_ds(train_arg, mode = 'train'))
        train_args = train_args[0]
        train_ds = ConcatDataset(train_ds)
    else: train_ds = get_ds(train_args, mode = 'train')

    # Validation Dataset Initialisation
    if type(val_args) == list:
        val_ds = []
        for val_arg in val_args:
            val_ds.append(get_ds(val_arg, mode = 'val'))
        val_ds = ConcatDataset(val_ds)
    else: val_ds = get_ds(val_args, mode = 'val')

    if not os.path.exists(f"{run_args.logs_fp}/train"):
        os.makedirs(f"{run_args.logs_fp}/train")
        os.makedirs(f"{run_args.logs_fp}/val")

    # --------------------------------------------------------------------------------------------

    # Training & Validation DataLoaders Initialisation
    train_dl = DataLoader(  train_ds, shuffle = False,
                            batch_size = train_args.batch_size,
                            num_workers = train_args.num_workers)
    val_dl = DataLoader(    val_ds, shuffle = False,
                            batch_size = val_args.batch_size,
                            num_workers = val_args.num_workers)
    if run_args.verbose: print(f"Training Samples: {len(train_ds)} | Validation Samples: {len(val_ds)}")
    
    # --------------------------------------------------------------------------------------------

    # 2D Segmentation Residual U-Net Initialisation
    model = ResidualUNet(train_args, run_args).to(run_args.device); criterion = dice_loss
    optimiser = torch.optim.Adam(model.parameters(), lr = run_args.resunet.lr_base)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                        mode = 'min', factor = run_args.resunet.lr_decay,
                        patience = run_args.resunet.lr_patience, verbose = True)
    best_val_loss = float('inf'); patience_counter = 0; best_model_state = None
    
    # ============================================================================================

    # Epoch Loop & Loss Initialisation
    full_train_loss, full_train_dice, full_train_combined = [], [], []
    full_val_loss, full_val_dice, full_val_combined = [], [], []; epoch = 0
    while epoch < run_args.resunet.num_epochs:

        # --------------------------------------------------------------------------------------------

        # Batch Loop | Training Step
        model.train(); train_loss, train_dice, train_combined = 0.0, 0.0, 0.0
        for train_idx, (ct, mask) in enumerate(train_dl):
            
            # Data Loading & Forward Pass
            ct, mask = ct.to(run_args.device), mask.to(run_args.device)
            optimiser.zero_grad()
            pred_mask = model(ct)

            # Loss Computation & Backpropagation | Training Step
            loss = criterion(pred_mask, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), run_args.resunet.grad_clip)
            optimiser.step()

            # --------------------------------------------------------------------------------------------

            # Metric Computation & Logging | Training Step
            train_loss += loss.item()
            with torch.no_grad():
                dice = dice_coef(pred_mask, mask).item()
                combined = combined_loss(pred_mask, mask).item()
                train_dice += dice; train_combined += combined
            
            # Metric Logging | Training Step
            if run_args.log_method == 'wandb' and run_logger is not None and (train_idx + 1) % run_args.loss_interval == 0:
                run_logger.log({"train/epoch": epoch,
                                "train/batch": train_idx,
                                "train/loss": loss.item(),
                                "train/dice": dice,
                                "train/combined": combined})
                print(f"Epoch {epoch+1}/{run_args.resunet.num_epochs} - Batch {train_idx + 1}/{len(train_dl)} - Loss: {loss.item():.4f}")
         
        # --------------------------------------------------------------------------------------------

        # Average Metric Computation & Logging | Training Step
        full_train_loss.append(train_loss / len(train_dl))
        full_train_dice.append(train_dice / len(train_dl))
        full_train_combined.append(train_combined / len(train_dl))
        if run_args.log_method == 'wandb' and run_logger is not None and epoch % run_args.log_interval == 0:
            run_logger.log({"train_avg/epoch": epoch,
                            "train_avg/loss": full_train_loss[-1],
                            "train_avg/dice": full_train_dice[-1],
                            "train_avg/combined": full_train_combined[-1]})
            
            # Mask Sample Logging | Training Step
            train_fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 3, 1); plt.axis('off'); plt.title('CT Scan')
            plt.imshow(ct[0, 0].detach().cpu().numpy(), cmap = 'gray')
            plt.subplot(1, 3, 2); plt.axis('off'); plt.title('Ground Truth Mask')
            plt.imshow(mask[0, 0].detach().cpu().numpy(), cmap = 'gray')
            plt.subplot(1, 3, 3); plt.axis('off'); plt.title('Predicted Mask')
            plt.imshow(pred_mask[0, 0].detach().cpu().numpy(), cmap = 'gray')
            run_logger.log({"train_avg/sample": wandb.Image(train_fig)})
            plt.savefig(f"{run_args.logs_fp}/train/sample_{epoch+1}.png")
            

        # ============================================================================================

        # Batch Loop | Validation Step
        model.eval(); val_loss, val_dice, val_combined = 0.0, 0.0, 0.0
        with torch.no_grad():
            for val_idx, (ct, mask) in enumerate(val_dl):
                
                # Data Loading & Forward Pass
                ct, mask = ct.to(run_args.device), mask.to(run_args.device)
                pred_mask = model(ct)

                # Loss Computation | Validation Step
                loss = criterion(pred_mask, mask)
                dice = dice_coef(pred_mask, mask).item()
                combined = combined_loss(pred_mask, mask).item()
                val_loss += loss.item(); val_dice += dice; val_combined += combined

                # --------------------------------------------------------------------------------------------

                # Metric Logging | Validation Step
                if run_args.log_method == 'wandb' and run_logger is not None and (val_idx + 1) % run_args.loss_interval == 0:
                    run_logger.log({"val/epoch": epoch,
                                    "val/batch": val_idx,
                                    "val/loss": loss.item(),
                                    "val/dice": dice,
                                    "val/combined": combined})
                    print(f"Epoch {epoch+1}/{run_args.resunet.num_epochs} - Val Batch {val_idx + 1}/{len(val_dl)} - Val Loss: {loss.item():.4f}")

        # Average Metric Computation & Logging | Validation Step
        full_val_loss.append(val_loss / len(val_dl))
        full_val_dice.append(val_dice / len(val_dl))
        full_val_combined.append(val_combined / len(val_dl))
        if run_args.log_method == 'wandb' and run_logger is not None and epoch % run_args.log_interval == 0:
            run_logger.log({"val_avg/epoch": epoch,
                            "val_avg/loss": full_val_loss[-1],
                            "val_avg/dice": full_val_dice[-1],
                            "val_avg/combined": full_val_combined[-1]})

            # Mask Sample Logging | Validation Step
            val_fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 3, 1); plt.axis('off'); plt.title('CT Scan')
            plt.imshow(ct[0, 0].detach().cpu().numpy(), cmap = 'gray')
            plt.subplot(1, 3, 2); plt.axis('off'); plt.title('Ground Truth Mask')
            plt.imshow(mask[0, 0].detach().cpu().numpy(), cmap = 'gray')
            plt.subplot(1, 3, 3); plt.axis('off'); plt.title('Predicted Mask')
            plt.imshow(pred_mask[0, 0].detach().cpu().numpy(), cmap = 'gray')
            run_logger.log({"val_avg/sample": wandb.Image(val_fig)})
            plt.savefig(f"{run_args.logs_fp}/val/sample_{epoch+1}.png")
        if run_args.verbose: print(f"Epoch {epoch+1}/{run_args.resunet.num_epochs} | Train Loss: {full_train_loss[-1]:.4f} | Val Loss: {full_val_loss[-1]:.4f}")
            
        # ============================================================================================

        # Learning Rate Scheduler Step
        lr_scheduler.step(full_val_loss[-1])
        current_lr = optimiser.param_groups[0]['lr']

        # Early Stopping8 Checkpoint
        if full_val_loss[-1] < best_val_loss:
            best_val_loss = full_val_loss[-1]; patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_args.logs_fp, f"{run_args.runV}_best.ckpt"))
            if run_args.verbose: print(f"Epoch {epoch+1}: Best model saved with Val Loss: {full_val_loss[-1]:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= run_args.resunet.es_patience:
                if run_args.verbose: print(f"Epoch {epoch+1}: Early stopping triggered after {patience_counter} epochs without improvement.")
                break
        
        # --------------------------------------------------------------------------------------------

        # Periodic Model Checkpoint
        if epoch % run_args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(run_args.logs_fp, f"{run_args.runV}_latest.ckpt"))
            if run_args.verbose: print(f"Epoch {epoch+1}: Model checkpoint saved.")
        epoch += 1

    # Metric Saving
    loss_history = {'train_loss': full_train_loss,
                    'val_loss': full_val_loss,
                    'train_dice': full_train_dice,
                    'val_dice': full_val_dice,
                    'train_combined': full_train_combined,
                    'val_combined': full_val_combined,
                    'best_val_loss': best_val_loss}
    np.save(f"{run_args.logs_fp}/loss_history.npy", loss_history)