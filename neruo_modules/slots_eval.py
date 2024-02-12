import copy
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
from SHAPES import SHAPESDATASET



import torchvision
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import random



from slots import SlotAutoencoder
import utils

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        batch_size, num_elements, input_size = x.size()
        x = x.view(-1, input_size)

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        out = out.view(batch_size, num_elements, -1)

        
        return out
    

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


    for _, (x, y) in loader:
        x = eval_slots(x.float())

        y = y.float()
        y = y.view(-1,y.size(-1))
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            y_hat = model(x)
            y = y.unsqueeze(0)
            cost_matrix = utils.calculate_distances(y,y_hat)
            h_match = utils.hungarian_algorithm(cost_matrix)
            loss = torch.sum(h_match[0])

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



# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.PILToTensor(),
#     ])
#     image = Image.open(image_path).convert('RGB')
#     input_tensor = transform(image)
    
#     input_batch = input_tensor.unsqueeze(0)

#     input_batch = (input_batch - 127.5) / 127.5
#     return input_batch

def to_numpy(x):
    return x.cpu().detach().numpy()

def renormalize(x):
    # x = x.clamp(min=-1, max=1)
    return x / 2. + 0.5  # [-1, 1] to [0, 1]

@torch.no_grad()
def get_prediction(model, batch, idx=0):
    recon_combined, recons, masks, slots = model(batch)
    image = renormalize(batch)[idx]
    recon_combined = renormalize(recon_combined)[idx]
    recons = renormalize(recons)[idx]
    masks = masks[idx]
    return image, recon_combined, recons, masks, slots



def eval_slots(data):

    model_path = "models/slotmodel.pt"

    ckpt = torch.load(model_path,map_location='cpu')

    model = SlotAutoencoder(
        in_shape=(3,64,64),
        width=32,
        num_slots=10,
        slot_dim=32,
        routing_iters=3,
    )

    model.load_state_dict(ckpt['model_state_dict'],)
    # model.cuda()

    # batch = preprocess_image(data)

    _, _, _, _, slots = get_prediction(model, data)

    return slots




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_dir", type=str, default="datasets/training_data")
    parser.add_argument("--max_num_obj", type=int, default=9)
    parser.add_argument("--input_res", type=int, default=64)
    # model
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--hidden_state", type=int, default=32)
    # training
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_known_args()[0]


    
    seed_all(args.seed, args.deterministic)
    os.makedirs(f"checkpoints/{args.exp_name}", exist_ok=True)
    h, w = int(args.input_res), int(args.input_res)

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

    model = Classifier(32,64,9)

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
   

    print(f"{model}\n#params: {sum(p.numel() for p in model.parameters()):,}")
    for k in sorted(vars(args)):
        print(f"--{k}={vars(args)[k]}")

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_loss = 1e6
    print("\nRunning sanity check...")
    _ = run_epoch(model, dataloaders["val"])

    for epoch in range(1, args.epochs):
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
                torch.save(save_dict, f"checkpoints/{args.exp_name}/{step}_ckpt.pt")








