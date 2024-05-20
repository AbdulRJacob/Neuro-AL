import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor



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
    

class ClassificationHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassificationHead, self).__init__()
        self.classification_head = nn.Sequential(
            nn.Linear(input_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classification_head(x)
    
class MLPHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPHead, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(input_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp_head(x)


class SlotAutoencoder(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        width: int = 64,
        num_slots: int = 10,
        slot_dim: int = 64,
        routing_iters: int = 3,
        classes: dict = {"shape": 3, "colour": 3, "size": 2},
        obj_info: dict = {"coords": 2, "real": 1}
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

        ## Multiple classification head
        self.classification_heads = nn.ModuleList([
            ClassificationHead(width, config) for config in list(classes.values())
        ])  

         ## Multiple MLP head
        self.mlp_heads = nn.ModuleList([
            MLPHead(width, config) for config in list(obj_info.values())
        ])      
     
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

        outputs = []
        for head in self.classification_heads:
            outputs.append(head(z).view(batch_size, num_elements, -1))

        outputs.insert(0,self.mlp_heads[0](z).view(batch_size, num_elements, -1))
        outputs.append(self.mlp_heads[1](z).view(batch_size, num_elements, -1))


        output = torch.cat(outputs, dim=2)


        return recon_combined, recons, masks, slots, output

