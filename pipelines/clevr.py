import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from symbolic_modules.aba_framework import ABAFramework

from datasets.CLEVR import CLEVR
import neuro_modules.utils as utils
from neuro_modules.NAL import NAL
from neuro_modules.slots import SlotAutoencoder
from neuro_modules import shapes_metics 
import neuro_modules.evaluation as e


def get_class_dict(object_dict : dict):
    length_dict = {}

    for key, value in object_dict.items():
        length_dict[key] = len(value)

    return length_dict


def clevr():
    object_info = {"object" : ["", "cube", "sphere", "cylinder"],
                   "colour": ["", "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
                   "size": ["", "large","small"],
                   "material" : ["","rubber","metal"]
                   }
    
    num_slots = 11
    
    ckpt = torch.load(os.getcwd() + "/models/clevr_m1.pt",map_location='cpu')

    model = SlotAutoencoder(
            in_shape=(3,64,64),
            classes=get_class_dict(object_info),
            width=32,
            num_slots=num_slots,
            slot_dim=32,
            routing_iters=3,
        )

    model.load_state_dict(ckpt['model_state_dict'],)
    data = "/mnt/d/fyp/CLEVR_v1.0/CLEVR_v1.0/images/test/CLEVR_test_000002.png"


    clevr_NAL = NAL(model, object_info)
    preds = clevr_NAL.run_slot_attention_model(data,num_slots)
    
    print(preds)

    e.visualise_slots(model,data,11)


        

if __name__ == "__main__":  
    clevr()
    


