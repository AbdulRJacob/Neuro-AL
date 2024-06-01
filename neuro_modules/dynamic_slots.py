from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import neuro_modules.slots as static_slots

class AdaptiveSlotAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        max_slots: int = 7,
        slot_dim: int = 64,
        routing_iters: int = 3,
        hidden_dim: int = 128,
        threshold: float = 0.1
    ):
        super().__init__()
        self.max_slots = max_slots
        self.slot_dim = slot_dim
        self.routing_iters = routing_iters
        self.threshold = threshold

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

    def forward(self, x: torch.Tensor, num_slots: Optional[int] = None):
        b = x.shape[0]
        x = self.ln_inputs(x)
        k = torch.einsum("bnc,cd->bnd", x, self.W_k)
        v = torch.einsum("bnc,cd->bnd", x, self.W_v)
        slots = self.loc + self.logscale.exp() * torch.randn(
            b, self.max_slots, self.slot_dim, device=x.device
        )

        for _ in range(self.routing_iters):
            slots_prev = slots
            slots = self.ln_slots(slots)
            q = torch.einsum("bkd,dd->bkd", slots, self.W_q) * self.slot_dim**-0.5
            agreement = torch.einsum("bkd,bdn->bkn", q, k.transpose(-2, -1))
            attn = agreement.softmax(dim=1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bkn,bnd->bkd", attn, v)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(b, -1, self.slot_dim)
            slots = slots + self.mlp(slots)

        # Use Gumbel-Softmax to determine the active slots
        mean_attention = attn.mean(dim=2)
        gumbel_logits = torch.randn_like(mean_attention)
        slot_weights = nn.functional.gumbel_softmax(gumbel_logits, hard=True, dim=1)
        active_slots = slot_weights.unsqueeze(2) * slots
        active_slots = active_slots.sum(dim=1)

        return active_slots



class AdaptiveSlotAutoencoder(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        width: int = 64,
        max_slots: int = 10,
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
            static_slots.PositionEmbed(width, in_shape[1:]),
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

        self.slot_attention = AdaptiveSlotAttention(
                input_dim=width,
                max_slots=max_slots,
                slot_dim=slot_dim,
                routing_iters=routing_iters,
                hidden_dim=width,
            )

        self.slot_grid = (in_shape[1] // 16, in_shape[2] // 16)
        dec_act = nn.LeakyReLU()
        self.decoder = nn.Sequential(
            static_slots.PositionEmbed(slot_dim, self.slot_grid),
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
            ),
        )

        self.classification_heads = nn.ModuleList([
            static_slots.ClassificationHead(width, config) for config in list(classes.values())
        ])  

        self.mlp_heads = nn.ModuleList([
            static_slots.MLPHead(width, config) for config in list(obj_info.values())
        ])      
     
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = self.encoder(x)
        x = self.mlp(x.reshape(*x.shape[:2], -1).permute(0, 2, 1))

        slots, slot_weights = self.slot_attention(x)

        # Here, we use slot_weights to mask the slots if needed
        active_slot_mask = slot_weights.sum(dim=1) > 0.5
        active_slots = slots * active_slot_mask.unsqueeze(2)

        x = active_slots.view(-1, active_slots.shape[-1])[:, :, None, None]
        x = x.repeat(1, 1, *self.slot_grid)

        x = self.decoder(x)
        x = x.view(b, -1, c + 1, h, w)
        recons, masks = torch.split(x, [c, 1], dim=2)
        masks = masks.softmax(dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)

        z = active_slots.detach()
        batch_size, num_elements, input_size = z.size()
        z = z.view(-1, input_size)

        outputs = []
        for head in self.classification_heads:
            outputs.append(head(z).view(batch_size, num_elements, -1))

        outputs.insert(0, self.mlp_heads[0](z).view(batch_size, num_elements, -1))
        outputs.append(self.mlp_heads[1](z).view(batch_size, num_elements, -1))

        output = torch.cat(outputs, dim=2)

        return recon_combined, recons, masks, active_slots, output

    

