import copy
import os
import random
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
from datasets.CLEVR import CLEVR
import neuro_modules.utils as utils
from neuro_modules.slots import SlotAutoencoder
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
    model: nn.Module, dataloader: DataLoader, optimizer: Optional[Optimizer] = None
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
        y = y.cuda()
        x = x.cuda()
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            recon_combined , _ ,_ ,_, y_hat = model(x)
            
            cost_matrix = utils.calculate_distances(y,y_hat,args.num_slots)

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
    import argparse

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_dir", type=str, default="../datasets/CLEVR/")
    parser.add_argument("--max_num_obj", type=int, default=9)
    parser.add_argument("--input_res", type=int, default=64)
    # model
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num_slots", type=int, default=10)
    parser.add_argument("--slot_dim", type=int, default=32)
    parser.add_argument("--routing_iters", type=int, default=3)
    # training
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_known_args()[0]

    seed_all(args.seed, args.deterministic)
    os.makedirs(f"./checkpoints/{args.exp_name}", exist_ok=True)

    n = 1
    h, w = int(args.input_res * n), int(args.input_res * n)
    aug = {
        "train": transforms.Compose(
            [
                transforms.Resize((h, w), antialias=None),
                # transforms.CenterCrop(args.input_res),
                # transforms.RandomCrop(args.input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((h, w), antialias=None),
                # transforms.CenterCrop(args.input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        ),
    }

    datasets = {
        split: CLEVR(
            data_dir=args.data_dir,
            transform=aug[split],
            cache=True,
        )
        for split in ["train", "val"]
    }

    datasets["test"] = copy.deepcopy(datasets["val"])

    kwargs = {
        "batch_size": args.batch_size,
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
        in_shape=(3, args.input_res, args.input_res),
        width=args.width,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        routing_iters=args.routing_iters,
        classes= {"shape": 4,"color": 9, "size": 3, "material": 3}
    ).cuda()


    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    checkpoint_path = os.getcwd() + '/checkpoints/default/ckpt.pt'

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
    for k in sorted(vars(args)):
        print(f"--{k}={vars(args)[k]}")


    if os.path.exists(checkpoint_path):
        optimizer = AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

    best_loss = 1e6

    # Configure logging
    log_file_path = "./logs/training_log_clevr.txt"
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as file:
            file.write("Log file created\n")
            print("Log file created at:", log_file_path)
    else:
        print("Log file already exists at:", log_file_path)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("\nRunning sanity check...")
    _ = run_epoch(model, dataloaders["val"])


    for epoch in range(1, args.epochs):
        print("\nEpoch {}:".format(epoch))
        train_loss  = run_epoch(model, dataloaders["train"], optimizer)

        if epoch % 4 == 0:
            valid_loss = run_epoch(model, dataloaders["val"])

            x, y = next(iter(dataloaders["val"]))

            x = (x / 127.5 ) - 1
            y = y.float()
            y = y.cuda()
            x = x.cuda()
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
                    "hparams": vars(args),
                    "epoch": epoch,
                    "step": step,
                    "best_loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(save_dict, f"./checkpoints/{args.exp_name}/{step}_ckpt.pt")

            logging.info("Epoch {}: Train loss: {:.6f}, Valid loss: {:.6f}".format(epoch, train_loss, valid_loss))
        else:
            logging.info("Epoch {}: Train loss: {:.6f}".format(epoch, train_loss))
