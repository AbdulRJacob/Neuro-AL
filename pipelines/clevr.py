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

from datasets.CLEVR.CLEVR import CLEVRHans
import neuro_modules.utils as utils
from neuro_modules.NAL import NAL
from neuro_modules.slots import SlotAutoencoder
import neuro_modules.evaluation as e

categories = {
    "object": ["cube", "sphere", "cylinder"],
    "colour": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "size": ["large", "small"],
    "material": ["rubber", "metal"]
}

def one_hot_encode_dict(data_dict):
    print(data_dict)
    one_hot_encoded = []
    for feature, values in categories.items():
        value = data_dict.get(feature)
        # One-hot encode the value
        if value in values:
            one_hot = [0] * len(values)
            one_hot[values.index(value)] = 1
            one_hot_encoded.extend(one_hot)
        else:
            one_hot_encoded.extend([0] * len(values))
    return one_hot_encoded

# Define a function to one-hot encode a list of dictionaries
def one_hot_encode_list(data_list):
    encoded_list = []
    for data_dict in data_list:
        encoded_list.append(one_hot_encode_dict(data_dict))
    return np.array(encoded_list)

def get_class_dict(object_dict : dict):
    length_dict = {}

    for key, value in object_dict.items():
        length_dict[key] = len(value)

    return length_dict


def get_clevr_nal_model():
    object_info = {"object" : ["cube", "sphere", "cylinder"],
                   "colour": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
                   "size": ["large","small"],
                   "material" : ["rubber","metal"]
                   }

    ckpt = torch.load(os.getcwd() + "/models/clevr_m1.pt",map_location='cpu')

    model = SlotAutoencoder(
            in_shape=(3,64,64),
            width=32,
            num_slots=11,
            slot_dim=32,
            routing_iters=3,
            classes= get_class_dict(object_info),
            obj_info={"coords": 3, "real": 1}
        )

    model.load_state_dict(ckpt['model_state_dict'],)

    nal = NAL(model,object_info)

    return nal


def clevr(num_examples: int):

    NUM_SLOTS = 11
    THRESHOLD = 0.9
   

    data = CLEVRHans()

    clevr_NAL = get_clevr_nal_model()
    postive_class = [0]
    negative_class = [1,2]
    print("choosing examples...")
    
    j=0
    history = []
    while j < num_examples:
        choosen_exp = random.randint(0, 3000 - 2)
        example = data[choosen_exp] 
        # print(example)

        if example["class"] in postive_class:
            prediction, _ = clevr_NAL.run_slot_attention_model(example["input"],NUM_SLOTS,num_coords=3)
            if clevr_NAL.check_prediction_quailty(prediction,THRESHOLD) and choosen_exp not in history:
                clevr_NAL.populate_aba_framework(prediction,True)
                j = j + 1
                history.append(choosen_exp)


    j=0
    history = []
    while j < num_examples:
        choosen_exp = random.randint(0, 3000 - 2)
        example = data[choosen_exp] 

        if example["class"] in negative_class:
            prediction, _ = clevr_NAL.run_slot_attention_model(example["input"],NUM_SLOTS,num_coords=3)
            if clevr_NAL.check_prediction_quailty(prediction,THRESHOLD) and choosen_exp not in history:
                clevr_NAL.populate_aba_framework(prediction,False)
                j = j + 1
                history.append(choosen_exp)


    filename = f"clevr_bk_c1.aba"
    print("running framework...")
    clevr_NAL.run_aba_framework(filename)


def clevr_nal_inference(img_path: str, aba_path: str, include_pos = False):

    nal = get_clevr_nal_model()

    NUM_SLOTS = 11

    prediction, _ = nal.run_slot_attention_model(img_path,NUM_SLOTS,num_coords=3)
    nal.populate_aba_framework_inference(prediction,include_pos)

    """
        Classification Task
        if predicate c(...) is in majority stable model we classify image as positve 
    
    """

    img_c0 = f":- not c(img_{nal.img_id - 1})."
    img_c1 = f":- not c2(img_{nal.img_id - 1})."
    img_c2 = f":- not c3(img_{nal.img_id - 1})."

    nal.load_framework(aba_path)
    
    c_models = nal.run_learnt_framework(img_c0)
    c2_models = nal.run_learnt_framework(img_c1)
    c3_models = nal.run_learnt_framework(img_c2)

    
    models = [len(c_models),len(c2_models),len(c3_models)]
    print(models)

    return np.argmax(np.array(models))

        

if __name__ == "__main__":  
    clevr(15)
    


