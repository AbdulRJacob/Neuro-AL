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

from datasets.SHAPES_4.SHAPES4 import SHAPESDATASET
import neuro_modules.utils as utils

from models.slots_shapes4 import SlotAutoencoder as SLOTS_4
from models.slots_shapes import SlotAutoencoder as SLOTS_9
from neuro_modules.NAL import NAL
from neuro_modules.slots import SlotAutoencoder
from neuro_modules import shapes_metics 


def get_SHAPES_dataset(data_dir, data_size, num_classes):

        values = []
        dataset = {}

        for i in range(1, num_classes + 1) :
            dataset[i] = []
            values = []
            for j in range(1,data_size +1):
                id = random.randint(0,data_size -1)
                while id in values:
                    id = random.randint(0, data_size -1)

                values.append(id)
                img_path = data_dir + f"c{i}_s{id}/c{i}_{id}.png"
                label = data_dir + f"c{i}_s{id}/labels.txt"
                dataset[i].append((img_path,label))

        return dataset

def get_ground_truth(label_path:str ):
        with open(label_path, "r") as file:
            lines = file.readlines()

            set_result = []
            for line in lines:
                item = line.strip().split(",") 
                if len(item) == 4: 
                    info = [int(item[0])]
                    info = info + item[1:]
                    set_result.append((tuple(info), 1))
                
            set_result.append(((0,'','',''),1))
             
            return set_result


def get_class_dict(object_dict : dict):
    length_dict = {}

    for key, value in object_dict.items():
        length_dict[key] = len(value)

    return length_dict

def get_shapes_bk():

    bk_rules = ["above(S1,S2,I) :- box(I,S1,X1,Y1,X2,Y2), box(I,S2,X12, Y12, X22,Y22), Y12 - Y2 > 0.",
                "below(S1,S2,I) :- box(I,S1,X1,Y1,X2,Y2), box(I,S2,X12, Y12, X22,Y22), Y1 - Y22 > 0.",
                "left(S1,S2,I) :- box(I,S1,X1,Y1,X2,Y2), box(I,S2,X12, Y12, X22,Y22), X2 - X12 < 0.",
                "right(S1,S2,I) :- box(I,S1,X1,Y1,X2,Y2), box(I,S2,X12, Y12, X22,Y22), X22 - X1 < 0."]
        
    return bk_rules

def shapes_4_nal_random():
    object_info = {"object" : ["","Triangle","Circle","Square"],
                   "colour": ["","Red","Green","Blue"]
                   }
    
    ckpt = torch.load(os.getcwd() + "/models/shapes4_m1.pt",map_location='cpu')

    model = SLOTS_4(
            in_shape=(3,64,64),
            width=32,
            num_slots=5,
            slot_dim=32,
            routing_iters=3,
        )

    model.load_state_dict(ckpt['model_state_dict'],)

    nal = NAL(model,object_info)

    NUM_EXAMPLES = 10
    DATA_DIR = "datasets/SHAPES_4/training_data/"
    NUM_CLASSESS = 10
    MAX_EXAMPLES = 200
    NUM_SLOTS = 5
    THRESHOLD = 0.85
    classes = [1,2]

    data = get_SHAPES_dataset(DATA_DIR,MAX_EXAMPLES,NUM_CLASSESS)

    for i in classes : ## range(1, SHAPES_CLASSESS + 1)
        j = 0
        history = []
        while j < NUM_EXAMPLES:
            choosen_exp = random.randint(0, MAX_EXAMPLES - 1)
            img_path = data[i][choosen_exp]  ## Tuple of (img_path, label_path)
            prediction = nal.run_slot_attention_model(img_path[0],NUM_SLOTS)

            if nal.check_prediction_quailty(prediction,THRESHOLD) and choosen_exp not in history:
                nal.populate_aba_framework(prediction, i == classes[0])
                j = j + 1
                history.append(choosen_exp)
    

    nal.run_aba_framework(id=NUM_EXAMPLES)

    pos_class = shapes_metics.calculate_aba_classification_accuracy(nal,str(classes[0]),"c")
    neg_class = shapes_metics.calculate_aba_classification_accuracy(nal,str(classes[1]),"c")

    print("Accuracy for Positive Examples: ", pos_class)
    print("Accuracy for Negative Examples: ", 1 - neg_class)
        


def shapes_4_nal_dissimlarity():

    object_info = {"object" : ["","Triangle","Circle","Square"],
                   "colour": ["","Red","Green","Blue"]
                   }
    

    ckpt = torch.load(os.getcwd() + "/models/shapes4_m1.pt",map_location='cpu')
    model = SLOTS_4( in_shape=(3,64,64), width=32, num_slots=5, slot_dim=32,routing_iters=3)
    model.load_state_dict(ckpt['model_state_dict'],)

    DATA_DIR = "datasets/SHAPES_4/training_data/"
    TRAINING_EXAMPLES = 100
    NUM_SLOTS = 5

    t = transforms.Compose([transforms.Resize((64, 64), antialias=None), 
                            transforms.PILToTensor()])

    dataset = SHAPESDATASET(data_dir=DATA_DIR,transform= t)

    data_loader = DataLoader(dataset, batch_size=TRAINING_EXAMPLES, num_workers=4)
    all_data = []

    for batch in data_loader:
        inputs, _ = batch
        all_data.append(inputs)


    nal = NAL(model,object_info)

    NUM_EXAMPLES = 20
    THRESHOLD = 0.75
    classes = [1,2]

    examples_c1 = utils.get_dissimilar_ranking(all_data[1],model,NUM_SLOTS)[:NUM_EXAMPLES]
    examples_c2 = utils.get_dissimilar_ranking(all_data[3],model,NUM_SLOTS)[:NUM_EXAMPLES]
    
    examples = [examples_c1,examples_c2]

    for i in range(len(classes)):
        for eg in examples[i]:
            img_path = DATA_DIR + f"c{classes[i]}_s{eg}/c{classes[i]}_{eg}.png"
            is_good = False
            while not is_good:
                prediction = nal.run_slot_attention_model(img_path,NUM_SLOTS)
                is_good = nal.check_prediction_quailty(prediction,THRESHOLD)

            nal.populate_aba_framework(prediction, i == 0)
    
    
    nal.run_aba_framework(id=NUM_EXAMPLES)

    pos_class = shapes_metics.calculate_aba_classification_accuracy(nal,str(classes[0]),"c")
    neg_class = shapes_metics.calculate_aba_classification_accuracy(nal,str(classes[1]),"c")

    print("Accuracy for Positive Examples: ", pos_class)
    print("Accuracy for Negative Examples: ", 1 - neg_class)


def shapes_nal():
    object_info = {"object" : ["","Triangle","Circle","Square"],
                   "colour": ["","Red","Green","Blue"],
                   "size:":  ["","Large","Small"]}

    ckpt = torch.load(os.getcwd() + "/models/shapes_m1.pt",map_location='cpu')

    model = SLOTS_9(
            in_shape=(3,64,64),
            width=32,
            num_slots=10,
            slot_dim=32,
            routing_iters=3,
        )

    model.load_state_dict(ckpt['model_state_dict'],)

    nal = NAL(model,object_info)

    images = ["datasets/SHAPES_9/training_data/c1_s1/c1_1.png",
              "datasets/SHAPES_9/training_data/c1_s16/c1_16.png"]
    
    for i in range(len(images)):
        pred = nal.run_slot_attention_model(images[i],10)
        nal.populate_aba_framework(pred,i,i == 1)
        nal.run_aba_framework(ground=True)
        

if __name__ == "__main__":  
    shapes_4_nal_random()
    


