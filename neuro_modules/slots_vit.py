import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple

import torch.optim as optim
from neuro_modules import slots
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers import  ViTModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from typing import Tuple

class DINO_ViT_Encoder(nn.Module):
    def __init__(self):
        super(DINO_ViT_Encoder, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    
    def forward(self, x):
        return self.vit(x)  # Patch features h

class MLPDecoder(nn.Module):
    def __init__(self, K, D_slots, N, D_feat):
        super(MLPDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D_slots, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, D_feat + 1)
        )
        self.K = K
        self.N = N
        self.D_feat = D_feat

    def forward(self, z):
        batch_size = z.size(0)
        z = z.view(-1, z.shape[-1])[:, None, :]
        z = z.repeat(1,self.N,1)
        z = self.mlp(z)  # shape (batch, K, D_feat)
        z = z.view(batch_size, self.K , self.N, self.D_feat + 1)
        # (b, num_slots, c, h, w), (b, num_slots, 1, h, w)
        recons, masks = torch.split(z, [self.D_feat, 1], dim=3)
        masks = masks.softmax(dim=1)
        # (b, c, h, w)
        recon_combined = torch.sum(recons * masks, dim=1)

        return recon_combined,masks

class DINOSlotAutoencoder(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        width: int = 64,
        num_slots: int = 10,
        slot_dim: int = 64,
        routing_iters: int = 3,
    ):
        super().__init__()
        self.encoder = DINO_ViT_Encoder()

        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

        self.patch_size = 16
        self.num_patches = (in_shape[1] // self.patch_size) * (in_shape[2] // self.patch_size)
        self.width = width

        self.slot_attention = slots.SlotAttention(
            input_dim=width,
            num_slots=num_slots,
            slot_dim=slot_dim,
            routing_iters=routing_iters,
            hidden_dim=width,
        )

        # Assuming the usage of MLPDecoder. Adjust parameters accordingly.
        self.decoder = MLPDecoder(K=num_slots, D_slots=slot_dim, N=self.num_patches, D_feat=768)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0) 
        h = self.encoder(x).last_hidden_state
        h = h[:, 1:, :] # remove cls
        # input_to_slot = self.mlp(h.reshape(*h.shape[:2], -1).permute(0, 2, 1))  # flatten img
        input_to_slots = h.view((batch_size,-1,self.width))
        
        slots = self.slot_attention(input_to_slots)
        ## atten shape is (b,k,n) -> (b,k,sqrt(n),sqrt(N)) -> (b,k,h,w) nearest neighbour 
        y, mask = self.decoder(slots)  # Reconstructed patch features
  
        return h, y, slots, mask
    
