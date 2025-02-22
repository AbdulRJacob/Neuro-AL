import copy
import os
import random
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from data.CLEVR import CLEVR
import utils.utils as utils
from models.slot_ae import SlotAutoencoder
import logging


def init_params(p):
    if isinstance(p, (nn.Linear, nn.Conv2d, nn.Parameter)):
        nn.init.xavier_uniform_(p.weight)
        p.bias.data.fill_(0)


def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_epoch(
    model: nn.Module, dataloader: DataLoader, gpu: bool = False, optimizer: Optional[Optimizer] = None
):
    training = False if optimizer is None else True
    model.train(training)
    loader = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        mininterval=(0.1 if os.environ.get("IS_NOHUP") is None else 60),
    )
    total_loss, count = 0, 0
    alpha = 0.7


    for _, (x, y) in loader:
        x = (x / 127.5 ) - 1
        y = y.float()
        y = y.cuda() if gpu else y
        x = x.cuda() if gpu else x
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            recon_combined , _ ,_ ,_, y_hat = model(x)
            
            cost_matrix = utils.calculate_distances(y,y_hat,num_slots)

            h_match = utils.hungarian_algorithm(cost_matrix)
            h_loss = torch.sum(h_match[0])

            r_loss = torch.mean((x - recon_combined) ** 2) 

            loss = alpha * h_loss + r_loss ## Tune Alpha so it comparable with r_loss

        if training:
            loss.backward()
            optimizer.step()
        count += x.shape[0]
        total_loss += loss.detach() * x.shape[0]
        loader.set_description(
            "=> {} | recon_cross_entropy_loss: {:.8f}".format(
                ("train" if training else "eval"), total_loss / count
            ),
            refresh=False,
        )
    return total_loss / count


if __name__ == "__main__":

    import yaml

    # Load the config.yaml
    with open("config/clevr_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    data_dir = config['data']['data_dir']
    max_num_obj = config['data']['max_num_obj']
    input_res = config['data']['input_res']

    width = config['model']['width']
    num_slots = config['model']['num_slots']
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
    
    seed_all(seed, deterministic)
    os.makedirs(f"./checkpoints/{exp_name}", exist_ok=True)

    n = 1
    h, w = int(input_res * n), int(input_res * n)
    aug = {
        "train": transforms.Compose(
            [
                transforms.Resize((h, w), antialias=None),
                # transforms.CenterCrop(input_res),
                # transforms.RandomCrop(input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((h, w), antialias=None),
                # transforms.CenterCrop(input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        ),
    }

    datasets = {
        split: CLEVR(
            data_dir=data_dir,
            transform=aug[split],
            cache=True,
        )
        for split in ["train", "val"]
    }

    datasets["test"] = copy.deepcopy(datasets["val"])

    kwargs = {
        "batch_size": batch_size,
        "num_workers": os.cpu_count(),  # 4 cores to spare
        "pin_memory": True,
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            **kwargs,
        )
        for split in ["train", "val", "test"]
    }

    model = SlotAutoencoder(
        in_shape=(3, input_res, input_res),
        width=width,
        num_slots=num_slots,
        slot_dim=slot_dim,
        routing_iters=routing_iters,
        classes= {"shape": 3,"color": 8, "size": 2, "material": 2},
        obj_info= {"coords": 3, "real": 1}

    )

    model = model.cuda() if gpu_present else model


    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    checkpoint_path = os.getcwd() + model_dir + '/checkpoints/default/ckpt.pt' 

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        step = checkpoint['step']
    else:
        model.apply(init_params)
        

    print(f"{model}\n#params: {sum(p.numel() for p in model.parameters()):,}")
    # Print config details
    for section, values in config.items():
        for key, value in values.items():
            print(f"--{key}={value}")


    if os.path.exists(checkpoint_path):
        optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    best_loss = 1e6

    # Configure logging
    log_file_path = logging_dir
    if not os.path.exists(log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'w') as file:
            file.write("Log file created\n")
            print("Log file created at:", log_file_path)
    else:
        print("Log file already exists at:", log_file_path)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("\nRunning sanity check...")
    _ = run_epoch(model, dataloaders["val"],gpu=gpu_present)


    for epoch in range(1, epochs):
        print("\nEpoch {}:".format(epoch))
        train_loss  = run_epoch(model, dataloaders["train"], gpu_present, optimizer)

        if epoch % 4 == 0:
            valid_loss = run_epoch(model, dataloaders["val"],gpu=gpu_present)

            x, y = next(iter(dataloaders["val"]))

            x = (x / 127.5 ) - 1
            y = y.float()
            y = y.cuda() if gpu_present else y
            x = x.cuda() if gpu_present else x
            _ , _ ,_ ,_, y_hat = model(x)
            step = int(epoch * len(dataloaders["train"]))

            ap = [utils.average_precision(y_hat.cpu().detach().numpy(), y.cpu().numpy(), d) for d in [-1., 1., 0.5, 0.25, 0.125]]
            logging.info(
                "Step {} | AP@inf: {:.2f}, AP@1: {:.2f}, AP@0.5: {:.2f}, AP@0.25: {:.2f}, AP@0.125: {:.2f}".format(step, ap[0], ap[1], ap[2], ap[3], ap[4])
            )
            logging.getLogger().handlers[0].flush()
    

            if valid_loss < best_loss:
                best_loss = valid_loss
                step = int(epoch * len(dataloaders["train"]))
                save_dict = {
                    "hparams": config,
                    "epoch": epoch,
                    "step": step,
                    "best_loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(save_dict, f"./checkpoints/{exp_name}/{step}_ckpt.pt")

            logging.info("Epoch {}: Train loss: {:.6f}, Valid loss: {:.6f}".format(epoch, train_loss, valid_loss))
        else:
            logging.info("Epoch {}: Train loss: {:.6f}".format(epoch, train_loss))
