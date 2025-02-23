import copy
import os
import random
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import optuna  # Import Optuna for hyperparameter tuning
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml

from data.SHAPES import SHAPESDATASET
import utils.utils as utils
from models.slot_ae import SlotAutoencoder, cluster_slots

# Initialize logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open("config/shapes_config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Hyperparameters from config
data_dir = config['data']['data_dir']
input_res = config['data']['input_res']

width = config['model']['width']
num_slots = config['model']['num_slots']  # Keep num_slots fixed
slot_dim = config['model']['slot_dim']
routing_iters = config['model']['routing_iters']

exp_name = config['training']['exp_name']
seed = config['training']['seed']
epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
deterministic = config['training']['deterministic']
logging_dir = config['training']['logging_dir']
model_dir = config['training']['model_dir']
gpu_present = config['training']['gpu']

# Set random seeds
def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

seed_all(seed, deterministic)

# Create directories
os.makedirs(f"./checkpoints/{exp_name}", exist_ok=True)

# Data augmentation
h, w = int(input_res), int(input_res)
aug = {
    "train": transforms.Compose([
        transforms.Resize((h, w), antialias=None),
        transforms.PILToTensor(),  
    ]),
    "val": transforms.Compose([
        transforms.Resize((h, w), antialias=None),
        transforms.PILToTensor(),  
    ]),
}

# Load datasets
datasets = {
    split: SHAPESDATASET(data_dir=data_dir, transform=aug[split], cache=True)
    for split in ["train", "val"]
}
datasets["test"] = copy.deepcopy(datasets["val"])

# Dataloader configurations
kwargs = {"batch_size": batch_size, "num_workers": os.cpu_count(), "pin_memory": True}
dataloaders = {split: DataLoader(datasets[split], shuffle=(split == "train"), drop_last=(split == "train"), **kwargs) for split in ["train", "val", "test"]}

# Training function
def run_epoch(model, alpha, dataloader, gpu=False, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loader = tqdm(enumerate(dataloader), total=len(dataloader), mininterval=0.1)
    total_loss, count = 0, 0

    for _, batch in loader:
        x, y = batch['input'], batch['target']
        x = (x / 127.5) - 1  
        y = y.float()
        y, x = (y.cuda(), x.cuda()) if gpu else (y, x)

        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            recon_combined, _, _, _, y_hat = model(x)
            cost_matrix = utils.calculate_distances(y, y_hat, num_slots)
            h_match = utils.hungarian_algorithm(cost_matrix)
            h_loss = torch.sum(h_match[0])
            r_loss = torch.mean((x - recon_combined) ** 2)
            loss = alpha * h_loss + r_loss  

        if training:
            loss.backward()
            optimizer.step()

        count += x.shape[0]
        total_loss += loss.detach() * x.shape[0]
        loader.set_description(f"=> {'train' if training else 'eval'} | loss: {total_loss / count:.6f}", refresh=False)
    
    return total_loss / count

# Optuna objective function
def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    slot_dim = trial.suggest_int("slot_dim", 32, 128, step=32)
    alpha = trial.suggest_uniform("alpha", 0.5, 1.0)  

    # Keep num_slots fixed
    config['training']['learning_rate'] = learning_rate
    config['training']['weight_decay'] = weight_decay
    config['training']['batch_size'] = batch_size
    config['model']['slot_dim'] = slot_dim

    # Update dataloaders with new batch size
    kwargs["batch_size"] = batch_size
    dataloaders = {split: DataLoader(datasets[split], shuffle=(split == "train"), drop_last=(split == "train"), **kwargs) for split in ["train", "val", "test"]}

    # Initialize model
    model = SlotAutoencoder(
        in_shape=(3, input_res, input_res),
        width=slot_dim,
        num_slots=num_slots,  
        slot_dim=slot_dim,
        routing_iters=routing_iters,
        classes={"shape": 3, "colour": 3, "size": 2}
    )
    model = model.cuda() if gpu_present else model

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss = float("inf")

    for epoch in range(epochs):
        train_loss = run_epoch(model, alpha, dataloaders["train"], gpu_present, optimizer)
        valid_loss = run_epoch(model, alpha, dataloaders["val"], gpu_present)

        logging.info(f"Trial {trial.number} | Epoch {epoch} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"./checkpoints/{exp_name}/best_model_trial_{trial.number}.pt")

        trial.report(valid_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_loss

# Run hyperparameter optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Log best hyperparameters
best_params = study.best_params
logging.info(f"Best Hyperparameters: {best_params}")
print("Best Hyperparameters:", best_params)
