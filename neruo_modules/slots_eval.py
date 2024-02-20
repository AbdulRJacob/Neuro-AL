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

from neruo_modules.slots import SlotAutoencoder


model_path = 'neruo_modules/checkpoints/default/26208_ckpt.pt'

ckpt = torch.load(model_path,map_location='cpu')

model = SlotAutoencoder(
        in_shape=(3,64,64),
        width=32,
        num_slots=10,
        slot_dim=32,
        routing_iters=3,
    )
model.load_state_dict(ckpt['model_state_dict'],)


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.PILToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    
    input_batch = input_tensor.unsqueeze(0)

    input_batch = (input_batch - 127.5) / 127.5
    return input_batch

def get_labels():
    all_labels = ["","Triangle","Circle","Square","Red","Green","Blue","L","S"] ## Change to three different vection 
    

    return {index: label for index, label in enumerate(list(all_labels))}

def get_predicted_symbols(indexes):
    ## Output a map of {slot: [position,shape,colour]}
    sym_table = {}

    for idx, info in enumerate(indexes):
        
        if 0 in info:
            sym_table[idx] = []
            continue

        sym_table[idx] = []
        for i in info:
            sym_table[idx].append(get_labels()[i])
        

    return sym_table

def get_prediction(image,idx=0):
    image = preprocess_image(image)
    model.eval()

    with torch.no_grad():
        _, _, _, _, output = model(image)
         

    
    sigmoid_output = output

    # print(sigmoid_output)
    prediction =  (sigmoid_output > 0.5).float().squeeze(0).numpy()
    indexes = [list(np.where(row == 1)[0]) for row in prediction]
    # print(indexes)

    print(get_predicted_symbols(indexes))

    return indexes
    