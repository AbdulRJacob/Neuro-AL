import copy
import os
import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from neuro_modules.slots import SlotAutoencoder


model_path = os.getcwd() + "/models/99216_ckpt.pt"
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


def to_numpy(x):
    return x.cpu().detach().numpy()

def renormalize(x):
    # x = x.clamp(min=-1, max=1)
    return x / 2. + 0.5  # [-1, 1] to [0, 1]

@torch.no_grad()
def get_visualisation (batch, idx=0):
    batch = preprocess_image(batch)
    recon_combined, recons, masks, slots, _  = model(batch)
    image = renormalize(batch)[idx]
    recon_combined = renormalize(recon_combined)[idx]
    recons = renormalize(recons)[idx]
    masks = masks[idx]
    
    image = to_numpy(image.permute(1,2,0))
    recon_combined = to_numpy(recon_combined.permute(1,2,0))
    recons = to_numpy(recons.permute(0,2,3,1))
    masks = to_numpy(masks.permute(0,2,3,1))
    slots = to_numpy(slots)


    num_slots = len(masks)
    fig, ax = plt.subplots(1, num_slots + 2, figsize=(18, 4))
    ax[0].imshow(image)
    ax[0].set_title('Input')
    ax[1].imshow(recon_combined)
    ax[1].set_title('Reconstruction')
    for i in range(num_slots):
        ax[i + 2].imshow(recons[i] * masks[i] + (1 - masks[i]))
        # ax[i + 2].imshow(recons[i] * masks[i])
        # ax[i + 2].imshow(masks[i])
        ax[i + 2].set_title('Slot %s' % str(i + 1))
    for i in range(len(ax)):
        ax[i].grid(False)
        ax[i].axis('off')

    plt.savefig('slot_vis.png')
    # plt.tight_layout()



def get_labels(shape,colour,size):
   shape_labels = ["","Triangle","Circle","Square"]
   colour_labels = ["","Red","Green","Blue"]
   size_labels = ["","L","S","M"]

   return shape_labels[shape], colour_labels[colour], size_labels[size]

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
    model.eval()

    with torch.no_grad():
        _, _, _, _ , output = model(image)

    
    out = output.squeeze(0)
    res =  []

    for i in range(10):
        pred = []
        item = [out[i][j] for j in range(3)]

        for k in item:
            pred.append(torch.argmax(k))


        res.append(pred)

    return res



def get_predicted_symbols(image):
    ## Output a map of {slot: [position,shape,colour]}
    sym_table = {}
    image  = preprocess_image(image)
    results = get_prediction(image)

    for idx,item in enumerate(results):
        shape = item[0].item()
        colour = item[1].item()
        size = item[2].item()


        shape, colour, size = get_labels(shape,colour,size)

        sym_table[idx] = [shape,colour,size]


    return sym_table

def get_true_symbols(labels):
    with open(labels, "r") as file:
        lines = file.readlines()

        sym_table = {}
        for line in lines:
            item = line.strip().split(",") 
            if len(item) == 4: 
                sym_table[int(item[0])] = item[1:]
        

        
        return sym_table

def get_classification_accuracy(data_path, dataset_size):
    ## Assume data_path is a directory of folders of the form 
    ## sample_data
    ##   |---img.png
    ##   |---labels.txt

    total_slot = 0
    total_correct_slot = 0
    total_correct_img = 0

    class_id = 1

    for i in range(1,dataset_size):

        k = i % 100
        offset = 900

        if (k == 0):
            class_id+=1
     
        
    
        img_path = data_path + f"c{class_id}_s{k+offset}/c{class_id}_{k+offset}.png"
        label_path = data_path + f"c{class_id}_s{k+offset}/labels.txt"

        pred_slots = get_predicted_symbols(img_path)
        true_slots = get_true_symbols(label_path)

        if i == 1:
            get_visualisation(img_path)
            print("Predicted:")
            print(pred_slots)
            print("Ground Truth: ")
            print(true_slots)

        same = all([set(pred_slots[key]) == set(true_slots[key]) for key in true_slots])


        if same:
            total_correct_img + 1

        for j in range(0,9):
            if pred_slots[j] == true_slots[j]:
                total_correct_slot+=1
            total_slot+=1

        
    print("-------Slot Classification Accuracy-----------")
    print("Slot Classification: ", total_correct_slot / total_slot)
    print("Full Image Accuracy: ", total_correct_img / dataset_size)

    










    
    