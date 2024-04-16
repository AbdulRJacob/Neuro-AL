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
from datasets.SHAPES.SHAPES import SHAPESDATASET
import neuro_modules.utils as utils
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


class SlotAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        num_slots: int = 7,
        slot_dim: int = 64,
        routing_iters: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.routing_iters = routing_iters

        self.ln_inputs = nn.LayerNorm(input_dim)
        self.ln_slots = nn.LayerNorm(self.slot_dim)

        self.W_q = nn.Parameter(torch.rand(self.slot_dim, self.slot_dim))
        self.W_k = nn.Parameter(torch.rand(input_dim, self.slot_dim))
        self.W_v = nn.Parameter(torch.rand(input_dim, self.slot_dim))
        self.loc = nn.Parameter(torch.zeros(1, self.slot_dim))
        self.logscale = nn.Parameter(torch.zeros(1, self.slot_dim))

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x: Tensor, num_slots: Optional[int] = None):
        # b: batch_size, n: num_inputs, c: input_dim, K: num_slots, d: slot_dim
        b = x.shape[0]
        # (b, n, c)
        x = self.ln_inputs(x)
        # (b, n, d)
        k = torch.einsum("bnc,cd->bnd", x, self.W_k)
        v = torch.einsum("bnc,cd->bnd", x, self.W_v)
        # (b, k, d)
        K = num_slots if num_slots is not None else self.num_slots
        slots = self.loc + self.logscale.exp() * torch.randn(
            b, K, self.slot_dim, device=x.device
        )

        for _ in range(self.routing_iters):
            slots_prev = slots
            slots = self.ln_slots(slots)
            # (b, k, d)
            q = torch.einsum("bkd,dd->bkd", slots, self.W_q)
            q = q * self.slot_dim**-0.5
            # (b, k, n)
            agreement = torch.einsum("bkd,bdn->bkn", q, k.transpose(-2, -1))
            attn = agreement.softmax(dim=1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean
            # (b, k, d)
            updates = torch.einsum("bkn,bnd->bkd", attn, v)
            # (b*k, d)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            )
            # (b, k, d)
            slots = slots.reshape(b, -1, self.slot_dim)
            slots = slots + self.mlp(slots)
        return slots



class PositionEmbed(nn.Module):
    def __init__(self, out_channels: int, resolution: Tuple[int, int]):
        super().__init__()
        # (1, height, width, 4)
        self.register_buffer("grid", self.build_grid(resolution))
        self.mlp = nn.Linear(4, out_channels)  # 4 for (x, y, 1-x, 1-y)

    def forward(self, x: Tensor):
        # (1, height, width, out_channels)
        grid = self.mlp(self.grid)
        # (batch_size, out_channels, height, width)
        return x + grid.permute(0, 3, 1, 2)

    def build_grid(self, resolution: Tuple[int, int]) -> Tensor:
        xy = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        xx, yy = torch.meshgrid(xy, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        grid = grid.unsqueeze(0)
        return torch.cat([grid, 1.0 - grid], dim=-1)


class SlotAutoencoder(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        width: int = 64,
        num_slots: int = 10,
        slot_dim: int = 64,
        routing_iters: int = 3,
        classes: dict = {"position": 9, "shape": 4,"colour": 4, "size": 3}
    ):
        super().__init__()
        enc_act = nn.ReLU()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[0], width, 5, padding=2),
            enc_act,
            *[nn.Conv2d(width, width, 5, padding=2), enc_act] * 3,
            PositionEmbed(width, in_shape[1:]),
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

        self.slot_attention = SlotAttention(
                input_dim=width,
                num_slots=num_slots,
                slot_dim=slot_dim,
                routing_iters=routing_iters,
                hidden_dim=width,
            )

        self.slot_grid = (in_shape[1] // 16, in_shape[2] // 16)
        dec_act = nn.LeakyReLU()
        self.decoder = nn.Sequential(
            PositionEmbed(slot_dim, self.slot_grid),
            *[
                nn.ConvTranspose2d(
                    width, width, 5, stride=2, padding=2, output_padding=1
                ),
                dec_act,
            ]
            * 4,
            nn.ConvTranspose2d(width, width, 5, stride=1, padding=2),
            dec_act,
            nn.ConvTranspose2d(
                width, in_shape[0] + 1, 3, stride=1, padding=1
            ),  # 4 output channels
        )

        self.classification_head_shape = nn.Sequential(
            nn.Linear(width, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, classes["shape"]),
            nn.Softmax()
        )

        self.classification_head_colour = nn.Sequential(
            nn.Linear(width, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, classes["colour"]),
            nn.Softmax()
        )

        self.classification_head_size = nn.Sequential(
            nn.Linear(width, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, classes["size"]),
            nn.Softmax()
        )

        ## Multiple classification head

     
    def forward(self, x: Tensor):
        # b: batch_size, c: channels, h: height, w: width, d: out_channels
        b, c, h, w = x.shape
        # (b, d, h, w)
        x = self.encoder(x)
        # (b, h*w, d)
        x = self.mlp(x.reshape(*x.shape[:2], -1).permute(0, 2, 1))  # flatten img

        # (b, num_slots, slot_dim)
        slots = self.slot_attention(x)

        # (b*num_slots, slot_dim, init_h, init_w)
        x = slots.view(-1, slots.shape[-1])[:, :, None, None]
        x = x.repeat(1, 1, *self.slot_grid)

       
        # (b*num_slots, c + 1, h, w)
        x = self.decoder(x)

        # (b, num_slots, c + 1, h, w)
        x = x.view(b, -1, c + 1, h, w)
        # (b, num_slots, c, h, w), (b, num_slots, 1, h, w)
        recons, masks = torch.split(x, [c, 1], dim=2)
        masks = masks.softmax(dim=1)
        # (b, c, h, w)
        recon_combined = torch.sum(recons * masks, dim=1)

        z = slots.detach()
        batch_size, num_elements, input_size = z.size()
        z = z.view(-1, input_size)

        z_colour = self.classification_head_colour(z)
        z_colour = z_colour.view(batch_size, num_elements, -1)
        z_shape = self.classification_head_shape(z)
        z_shape = z_shape.view(batch_size, num_elements, -1)
        z_size = self.classification_head_size(z)
        z_size= z_size.view(batch_size, num_elements, -1)


        output = torch.cat((z_shape, z_colour,z_size), dim=2)


        return recon_combined, recons, masks, slots, output



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
    parser.add_argument("--data_dir", type=str, default="../datasets/SHAPES/training_data")
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
        split: SHAPESDATASET(
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
    print("\nRunning sanity check...")
    _ = run_epoch(model, dataloaders["val"])

    # Configure logging
    log_file_path = "./logs/training_log.txt"
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    for epoch in range(1, args.epochs):
        logging.info("\nEpoch {}:".format(epoch))
        print("\nEpoch {}:".format(epoch))
        train_loss = run_epoch(model, dataloaders["train"], optimizer)

        if epoch % 4 == 0:
            valid_loss = run_epoch(model, dataloaders["val"])

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
