# Library Imports
import os
import random
import json
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

# Function Imports
from pathlib import Path

# ============================================================================================

def nest_args(flat_dict):
    nested = {}
    for key, value in flat_dict.items():
        if "." in key:
            group, subkey = key.split(".", 1)
            nested.setdefault(group, {})[subkey] = value
        else:
            nested[key] = value
    return nested

def dict_to_namespace(d):
    from argparse import Namespace
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return Namespace(**d)

# ============================================================================================

# Run Arguments Initialisation
def run_parser(
    model: str = 'fedseg',
    runV: str = 'VI',
    save: bool = False,
):  
    
    # Run Fundamentals
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = model)
    parser.add_argument('--runV', type = str, default = runV)
    parser.add_argument('--verbose', type = bool, default = True)
    parser.add_argument('--base_fp', type = str,
                        default = f"/nas-ctm01/homes/pfsousa")
    args = parser.parse_args("")

    # --------------------------------------------------------------------------------------------

    # Load Existing Arguments if Available
    save_fp = f"{args.base_fp}/{args.model}/runs/args_{args.runV}.yaml"
    if Path(save_fp).exists():
        if args.verbose: print(f"Loading ARGUMENT PARSER | {save_fp}")
        with open(Path(save_fp), "r") as f: args = dict_to_namespace(yaml.safe_load(f))
    else:

    # ============================================================================================

        # Directory Arguments
        parser.add_argument('--args_fp', type = str, default = save_fp)
        parser.add_argument('--script_fp', type = str,
                            default = f"{args.base_fp}/{args.model}")
        parser.add_argument('--logs_fp', type = str,
                            default = f"{args.base_fp}/{args.model}/logs/run_{args.runV}")
        
        # --------------------------------------------------------------------------------------------

        # Result Logging Arguments 
        parser.add_argument('--num_gpu', type = int, default = 1)
        parser.add_argument('--save_interval', type = int, default = 500)
        parser.add_argument('--loss_interval', type = int, default = 1)
        parser.add_argument('--log_interval', type = int, default = 100)
        parser.add_argument('--save_img', type = int, default = 2)
        parser.add_argument('--log_method', type = str,
                            choices = {'wandb', 'tensorboard', None},
                            default = 'wandb')
        
        # ============================================================================================

        # Architecture Fundamentals Arguments
        parser.add_argument('--seed', type = int, default = 1234)
        #parser.add_argument('--dim', type = int, default = 64)
        #parser.add_argument('--num_channel', type = int, default = 1)

        # ============================================================================================

        # ResidualUNet Architecture Arguments | Run Basics
        parser.add_argument('--resunet.resume', type = bool, default = False)
        

        # ============================================================================================

        # Argument File Saving
        args = parser.parse_args("")
        if save:
            if args.verbose: print(f"Saving ARGUMENT PARSER | {save_fp}")
            if not Path(save_fp).parent.exists(): os.makedirs(Path(save_fp).parent)
            with open(Path(save_fp), "w") as f:
                yaml.safe_dump(nest_args(vars(args)), f, sort_keys = False)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    return args

# ============================================================================================