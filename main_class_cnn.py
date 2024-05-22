import os
# Set up environment variables
os.environ["WANDB_API_KEY"] = "5f04d2ce100707f23b71379f67f28901d496edda"
os.environ["WANDB_MODE"] = "disabled"

import argparse
import yaml
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import copy

from dataset.dataloaders import get_dataset
from dataset.transforms import data_transform
from models.models import LinearModel_MNIST, LinearModel_CIFAR10, LinearModel_CIFAR100
from models.models_cnn import CNNModel, CNNModel_Pool
from utils.loss import C2Loss_Classification_CNN_Test, gradient_centralization

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info

# Argument Parser
parser = argparse.ArgumentParser(description='Autoencoder Training')
parser.add_argument('--yaml', type=str, default="./yaml/cnn/cifar10_ccl.yaml", help='Path to the YAML configuration file')
parser.add_argument('--batchsize', type=int, default=256, help='Batch size')
parser.add_argument('--act_F', type=str, default="elu", help='Forward learning rate')
parser.add_argument('--act_B', type=str, default="elu", help='Backward learning rate')
parser.add_argument('--lr_F', type=float, default=1., help='Forward learning rate')
parser.add_argument('--lr_B', type=float, default=1., help='Backward learning rate')
parser.add_argument('--fw_bn', type=int, default=0, help='Forward batch normalization')
parser.add_argument('--bw_bn', type=int, default=2, help='Backward batch normalization')
parser.add_argument('--mmt_F', type=float, default=0.9, help='Forward batch normalization')
parser.add_argument('--mmt_B', type=float, default=0.9, help='Backward batch normalization')
parser.add_argument('--grad_clip_F', type=float, default=0.5, help='Gradient clipping for forward pass')
parser.add_argument('--grad_clip_B', type=float, default=0.5, help='Gradient clipping for backward pass')
parser.add_argument('--warmup', type=int, default=200, help='Warmup steps')
parser.add_argument('--epochs', type=int, default=100, help='Cosine Annealing T_max')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--filter_target', type=float, default=0.2, help='Filter target')
parser.add_argument('--loss_scale_C', type=float, default=1, help='Filter target')
parser.add_argument('--loss_scale_ssl', type=float, default=0.5, help='Filter target')
args, _ = parser.parse_known_args()

with open(args.yaml, 'r') as file:
    yaml_config = yaml.safe_load(file)

# Set default arguments from YAML
parser.set_defaults(**yaml_config['method'])
parser.set_defaults(**yaml_config['dataset'])
parser.set_defaults(**yaml_config['training'])
args, _ = parser.parse_known_args()

logger.info(args)

# Set up wandb
run_name = f"{args.dataset}_{args.method}_{args.task_transform}"
wandb.init(project="opt-sigma-classfication-cnn", name=run_name, config=args)

args.device = device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
train_loader, val_loader, test_loader = get_dataset(args)

# Define the model
if args.architecture == "simple_linear" and args.dataset == "MNIST":
    model = LinearModel_MNIST(args).to(args.device)
elif args.architecture == "simple_linear" and args.dataset == "CIFAR10":
    model = LinearModel_CIFAR10(args).to(args.device)
elif args.architecture == "simple_linear" and args.dataset == "CIFAR100":
    model = LinearModel_CIFAR100(args).to(args.device)
elif args.architecture == "cnn":
    model = CNNModel(args).to(args.device)
elif args.architecture == "cnn_pool":
    model = CNNModel_Pool(args).to(args.device)

# Define the optimizer
forward_optimizer = optim.SGD(model.forward_params, lr=args.lr_F, momentum=args.mmt_F, weight_decay=args.wd_F)
backward_optimizer = optim.SGD(model.backward_params, lr=args.lr_B, momentum=args.mmt_B, weight_decay=args.wd_B)
if args.tmax !=0:
    forward_scheduler = CosineAnnealingLR(forward_optimizer, T_max=args.tmax, eta_min=args.eta_min)
    backward_scheduler = CosineAnnealingLR(backward_optimizer, T_max=args.tmax, eta_min=args.eta_min)

# Define the loss function
criterion = C2Loss_Classification_CNN_Test(args)
CELoss = nn.CrossEntropyLoss()

# Define fixed 
if args.dataset == "MNIST" or args.dataset == "CIFAR10" or args.dataset == "FashionMNIST" or args.dataset == "FashionMNIST_CNN" or args.dataset == "MNIST_CNN" or args.dataset == "STL10_cls": # from 0 - 9
    args.T10 = torch.Tensor([0,1,2,3,4,5,6,7,8,9]).long().to(device)
elif args.dataset == "CIFAR100": # from 0 - 99
    args.T10 = torch.Tensor([i for i in range(100)]).long().to(device)

# Training loop
best_val_loss = np.inf
args.train_steps = 0
for args.epoch in range(args.epochs):
    
    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        # if batch_idx > 200: continue
        data_fw, data_bw = data, args.T10
        if args.method == "CCL":
            activations = model(data_fw.to(args.device), detach_grad=True)
            signals = model.reverse(data_bw.to(args.device), detach_grad=True)
            loss, loss_item = criterion(activations, signals, target.to(args.device), method="local")
        elif args.method == "BP":
            activations = model(data_fw.to(args.device), detach_grad=False)
            loss = CELoss(activations[-1], target.to(args.device))
            loss_item = loss.item()
        if args.train_steps < args.warmup : 
            loss *= (batch_idx+1) / args.warmup
        forward_optimizer.zero_grad(), backward_optimizer.zero_grad()
        loss.backward()
        if args.GradC == 1: gradient_centralization(model)
        if args.grad_clip_F != 0: torch.nn.utils.clip_grad_norm_(model.forward_params, args.grad_clip_F)
        if args.grad_clip_B != 0: torch.nn.utils.clip_grad_norm_(model.backward_params, args.grad_clip_B)
        forward_optimizer.step(), backward_optimizer.step()
        args.train_steps += 1
    if args.tmax !=0:
        forward_scheduler.step(), backward_scheduler.step()

    # Validation loop
    model.eval()
    val_loss, val_loss_B, val_counter = 0, 0, 0
    val_acc = 0
    with torch.no_grad():
        for data, target in val_loader:
            data_fw, data_bw = data, args.T10
            if args.method == "CCL":
                activations = model(data_fw.to(args.device), detach_grad=True)
                loss_item = CELoss(activations[-1], target.to(args.device)).item()
            elif args.method == "BP":
                activations = model(data_fw.to(args.device), detach_grad=True)
                loss_item = CELoss(activations[-1], target.to(args.device)).item()
            val_counter += len(data)
            val_acc += (torch.argmax(activations[-1], dim=1) == target.to(args.device)).sum().item()
            val_loss += loss_item * len(data)

    # log loss
    wandb.log({"val_loss": val_loss / val_counter}, step=args.epoch)
    wandb.log({"val_acc": val_acc / val_counter}, step=args.epoch)

    logger.info(f"Epoch {args.epoch} | Val Loss: {val_loss / val_counter:.4f} | Val Acc: {val_acc / val_counter:.4f}")

    # log learning rate
    if args.method == "CCL":
        wandb.log({"lr_F": forward_optimizer.param_groups[0]['lr'], "lr_B": backward_optimizer.param_groups[0]['lr']}, step=args.epoch)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        args.best_epoch = args.epoch

# Test loop
model = best_model
model.eval()

test_loss, test_loss_B, test_counter = 0, 0, 0
test_acc = 0
with torch.no_grad():
    for data, target in test_loader:
        data_fw, data_bw = data, args.T10
        if args.method == "CCL":
            activations = model(data_fw.to(args.device), detach_grad=True)
            loss_item = CELoss(activations[-1], target.to(args.device)).item()
        elif args.method == "BP":
            activations = model(data_fw.to(args.device), detach_grad=True)
            loss_item = CELoss(activations[-1], target.to(args.device)).item()
        test_counter += len(data)
        test_loss += loss_item * len(data)
        test_acc += (torch.argmax(activations[-1], dim=1) == target.to(args.device)).sum().item()

wandb.log({"test_loss": test_loss / test_counter}, step=args.epoch)
wandb.log({"test_acc": test_acc / test_counter}, step=args.epoch)
wandb.log({"best_epoch": args.best_epoch}, step=args.epoch)
logger.info(f"Test Loss: {test_loss / test_counter:.4f} | Test Acc: {test_acc / test_counter:.4f}")