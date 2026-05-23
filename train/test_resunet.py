# ============================================================================================

# Package Import
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
import torch
#from seg_piu.metrics import loss_dice
import wandb
import pytorch_lightning as pl
#import hydra
import torch.distributed as dist
import tensorflow as tf
import cv2

# --------------------------------------------------------------------------------------------

# Functionality Import | Fundamentals
from pathlib import Path
from math import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, dataset
from datetime import datetime
#from omegaconf import DictConfig, open_dict

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
from keras_resunet import keras_residualUNet

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ============================================================================================

def compute_iou(pred, target, smooth=1e-6):
        """Compute Intersection over Union (IoU/Jaccard index)"""
        pred_binary = (pred > 0.5).float()
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)

# wrapper for your test_resunet.py
import os
import tempfile
import shutil

def safe_load_keras_model(remote_path, compile=False):
    """
    Load a Keras model safely from a shared cluster filesystem
    """
    # Disable HDF5 locking
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    
    try:
        # First attempt: try loading directly (maybe it works with locking off)
        import tensorflow as tf
        model = tf.keras.models.load_model(remote_path, compile=compile)
        print("✅ Loaded directly from shared filesystem")
        return model
    except Exception as e:
        print(f"Direct load failed: {e}")
        print("Attempting local copy method...")
        
        # Second attempt: copy to local temp
        # Use SLURM_TMPDIR if available, otherwise /tmp
        temp_dir = os.environ.get('SLURM_TMPDIR', '/tmp')
        local_path = os.path.join(temp_dir, f"temp_model_{os.getpid()}.keras")
        
        try:
            shutil.copy(remote_path, local_path)
            model = tf.keras.models.load_model(local_path, compile=compile)
            print(f"✅ Loaded from local copy: {local_path}")
            return model
        finally:
            # Clean up
            if os.path.exists(local_path):
                os.remove(local_path)

def postprocess(pred_mask, threshold=0.5, bottom_crop=0.12):
    """
    Full pipeline: remove table, separate lungs.
    """
    # Binarize
    mask = (pred_mask > threshold).astype(np.uint8) * 255
    
    # Remove bottom table
    h = mask.shape[0]
    crop_rows = int(h * bottom_crop)
    if crop_rows > 0:
        mask[-crop_rows:, :] = 0
    
    # Keep largest component in left and right halves separately
    w = mask.shape[1]
    mid = w // 2
    
    left = mask[:, :mid]
    right = mask[:, mid:]
    
    def largest_component(region):
        if np.sum(region) == 0:
            return region
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        if num_labels <= 1:
            return region
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        return (labels == largest_idx).astype(np.uint8) * 255
    
    left_clean = largest_component(left)
    right_clean = largest_component(right)
    
    # Combine
    final_mask = np.zeros_like(mask)
    final_mask[:, :mid] = left_clean
    final_mask[:, mid:] = right_clean
    
    # Optional: morphological close to fill small holes
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return final_mask

# ============================================================================================

def test_resunet(
    ds = None,
    run_args = None,
    run_logger = None
):

    # Dataset Initialisation
    data_args = ds.args
    if not os.path.exists(f"{run_args.test.save_fp}"):
        os.makedirs(f"{run_args.test.save_fp}")

    # --------------------------------------------------------------------------------------------

    # DataLoaders Initialisation
    dl = DataLoader(ds, shuffle = False,
                    batch_size = data_args.batch_size,
                    num_workers = data_args.num_workers)
    if run_args.verbose: print(f"Test Samples: {len(dl)}")
    
    # --------------------------------------------------------------------------------------------

    # 2D Segmentation Residual U-Net Loading
    model = ResidualUNet(data_args, run_args)
    model.load_state_dict(torch.load(run_args.test.resume_ckpt,
                                map_location = run_args.device))
    model.to(run_args.device); model.eval()

    # 
    #model = tf.keras.models.load_model( run_args.test.resume_ckpt,
    #                                    compile = False,
    #    custom_objects={'dice_coef': dice_coef, 'loss_dice': dice_loss})
    #model = safe_load_keras_model(run_args.test.resume_ckpt, compile=False)

    #
    #from tensorflow.keras.models import model_from_json
    #with open(f'{run_args.logs_fp}/model_architecture.json', 'r') as f:
    #    model_json = f.read()
    #model = model_from_json(model_json)
    #model.load_weights(f'{run_args.logs_fp}/m3_adebayo1.weights.h5')

    # ============================================================================================

    # Batch Loop | Testing Step
    whole_combined, whole_dice, whole_iou = 0.0, 0.0, 0.0
    whole_tp, whole_fp, whole_tn, whole_fn = 0.0, 0.0, 0.0, 0.0
    whole_sensitivity, whole_precision, whole_specificity = 0.0, 0.0, 0.0
    with torch.no_grad():
        for test_idx, (ct, mask) in enumerate(dl):
            full_combined, full_dice, full_iou = 0.0, 0.0, 0.0
            full_tp, full_fp, full_tn, full_fn = 0.0, 0.0, 0.0, 0.0
            full_sensitivity, full_precision, full_specificity = 0.0, 0.0, 0.0
            
            # 2D Data Loading & Forward Pass
            ct, mask = ct.to(run_args.device), mask.to(run_args.device)
            for slice_idx in range(ct.shape[1]):
                ct_slice = ct[:, slice_idx, :, :].unsqueeze(0)
                mask_slice = mask[:, slice_idx, :, :].unsqueeze(0)
                pred_mask = model(ct_slice).detach().cpu().numpy()
                pred_mask = torch.Tensor(postprocess(pred_mask[0, 0]))
                pred_mask = pred_mask.unsqueeze(0).unsqueeze(0).to(run_args.device)
                #pred_mask = model(ct_slice[0].unsqueeze(-1).numpy())
                #pred_mask = torch.Tensor(pred_mask.numpy()).permute(0, 3, 1, 2)
                pred_mask = (pred_mask > 0.5).float()

                # --------------------------------------------------------------------------------------------

                # Metric Computation
                comb_loss = combined_loss(pred_mask, mask_slice).item()
                dice = dice_coef(pred_mask, mask_slice).item()
                iou_loss = compute_iou(pred_mask, mask_slice).item()
                tp = (pred_mask * mask_slice).sum().item()
                fp = (pred_mask * (1 - mask_slice)).sum().item()
                tn = ((1 - mask_slice) * (1 - pred_mask)).sum().item()
                fn = (mask_slice * (1 - pred_mask)).sum().item()
                sensitivity = tp / (tp + fn + 1e-6)
                precision = tp / (pred_mask.sum() + 1e-6)
                specificity = tn / (tn + fp + 1e-6)
                
                # Metric Aggregation
                full_combined += comb_loss; full_dice += dice; full_iou += iou_loss
                full_tp += tp; full_fp += fp; full_tn += tn; full_fn += fn
                full_sensitivity += sensitivity; full_precision += precision; full_specificity += specificity
                
                # Metric Logging | Slice Level
                if run_args.log_method == 'wandb' and run_logger is not None:
                    run_logger.log({"test_slice/sample_idx": test_idx,
                                    "test_slice/slice_idx": slice_idx,
                                    "test_slice/combined": comb_loss,
                                    "test_slice/dice": dice,
                                    "test_slice/iou": iou_loss,
                                    "test_slice/tp": tp, "test_slice/fp": fp,
                                    "test_slice/tn": tn, "test_slice/fn": fn,
                                    "test_slice/sensitivity": sensitivity,
                                    "test_slice/precision": precision,
                                    "test_slice/specificity": specificity})

                # --------------------------------------------------------------------------------------------

                # Mask Sample Logging
                if run_logger is not None and test_idx % run_args.test.save_interval == 0:
                    fig = plt.figure(figsize=(30, 10))
                    plt.subplot(1, 3, 1); plt.axis('off'); plt.title('CT Scan')
                    plt.imshow(ct_slice[0, 0].detach().cpu().numpy(), cmap = 'gray')
                    plt.subplot(1, 3, 2); plt.axis('off'); plt.title('Ground Truth Mask')
                    plt.imshow(mask_slice[0, 0].detach().cpu().numpy(), cmap = 'gray')
                    plt.subplot(1, 3, 3); plt.axis('off'); plt.title('Predicted Mask')
                    plt.imshow(pred_mask[0, 0].detach().cpu().numpy(), cmap = 'gray')
                    run_logger.log({"test_slice/fig": wandb.Image(fig)})
                    #plt.savefig(f"{run_args.logs_fp}/test/sample_{epoch+1}.png")
                    fig.clear(); plt.close()
                if run_args.test.save_img and run_args.test.save_fp is not None:
                    pred_mask_np = pred_mask[0, 0].detach().cpu().numpy()
                    pred_mask_img = Image.fromarray((pred_mask_np * 255).astype(np.uint8))
                    pred_mask_img.save(f"{run_args.test.save_fp}/pred_{test_idx}_{slice_idx}.png")

            # Metric Aggregation
            whole_combined += full_combined / ct.shape[1]
            whole_dice += full_dice / ct.shape[1]
            whole_iou += full_iou / ct.shape[1]
            whole_tp += full_tp / ct.shape[1]
            whole_fp += full_fp / ct.shape[1]
            whole_tn += full_tn / ct.shape[1]
            whole_fn += full_fn / ct.shape[1]
            whole_sensitivity += full_sensitivity / ct.shape[1]
            whole_precision += full_precision / ct.shape[1]
            whole_specificity += full_specificity / ct.shape[1]

            # --------------------------------------------------------------------------------------------
            
            # Metric Logging | Full Subject Level
            if run_args.log_method == 'wandb' and run_logger is not None:
                run_logger.log({"test_subj/combined": full_combined / ct.shape[1],
                                "test_subj/dice": full_dice / ct.shape[1],
                                "test_subj/iou": full_iou / ct.shape[1],
                                "test_subj/tp": full_tp / ct.shape[1],
                                "test_subj/fp": full_fp / ct.shape[1],
                                "test_subj/tn": full_tn / ct.shape[1],
                                "test_subj/fn": full_fn / ct.shape[1],
                                "test_subj/sensitivity": full_sensitivity / ct.shape[1],
                                "test_subj/precision": full_precision / ct.shape[1],
                                "test_subj/specificity": full_specificity / ct.shape[1]})
    
    # Metric Logging | Whole Dataset Level
    if run_args.log_method == 'wandb' and run_logger is not None:
        run_logger.log({"test_ds/combined": whole_combined / len(dl),
                        "test_ds/dice": whole_dice / len(dl),
                        "test_ds/iou": whole_iou / len(dl),
                        "test_ds/tp": whole_tp / len(dl),
                        "test_ds/fp": whole_fp / len(dl),
                        "test_ds/tn": whole_tn / len(dl),
                        "test_ds/fn": whole_fn / len(dl),
                        "test_ds/sensitivity": whole_sensitivity / len(dl),
                        "test_ds/precision": whole_precision / len(dl),
                        "test_ds/specificity": whole_specificity / len(dl)})