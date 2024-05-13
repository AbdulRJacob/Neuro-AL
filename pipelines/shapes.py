import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.SHAPES_4.SHAPES4 import SHAPESDATASET
from datasets.SHAPES_4.SHAPES4 import SHAPES_4
from datasets.SHAPES_9.SHAPES import SHAPES
import neuro_modules.utils as utils

from models.slots_shapes4 import SlotAutoencoder as SLOTS_4
from models.slots_shapes import SlotAutoencoder as SLOTS_9
from neuro_modules.NAL import NAL
from neuro_modules import shapes_metics 
import neuro_modules.evaluation as e


def get_SHAPES_dataset(data_dir, data_size, num_classes, offset=0):
        dataset = {}

        for i in range(1, num_classes + 1) :
            dataset[i] = []
            for j in range(1+ offset ,offset + data_size):
                img_path = data_dir + f"c{i}_s{j}/c{i}_{j}.png"
                label = data_dir + f"c{i}_s{j}/labels.txt"
                dataset[i].append((img_path,label))

        return dataset

def get_ground_truth(label_path:str ):
        with open(label_path, "r") as file:
            lines = file.readlines()

            set_result = []
            for line in lines:
                item = line.strip().split(",") 
                if len(item) >= 3: 
                    info = [int(item[0])]
                    info = info + item[1:]
                    set_result.append((tuple(info)))
                
            set_result.append(((0,'','','')))
             
            return set_result
        
def get_ground_truth_map(label_path: str, dataset: int):
    labels = get_ground_truth(label_path)[:-1]
    generator = None

    if dataset == 4:
        generator = SHAPES_4(200,200,"")
    else:
        generator = SHAPES(300,300,"")

    map = generator.generate_attention_map(labels)

    map = cv2.resize(map, (64,64),interpolation=cv2.INTER_NEAREST)
    print(map.shape)

    return map


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

def get_rule(num_class: int):
    if num_class <= 2:
        return 1
    if num_class <= 4:
        return 2
    if num_class <= 6:
        return 3
    if num_class <= 8:
        return 4
    if num_class <= 10:
        return 5

def shapes_4_nal():
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
            ground = get_rule(i) in [3,4]
            if nal.check_prediction_quailty(prediction,THRESHOLD) and choosen_exp not in history:
                nal.populate_aba_framework(prediction, i == classes[0],include_pos=ground)
                j = j + 1
                history.append(choosen_exp)
    
    
    filename = "shapes_4_bk.aba"
    ground = get_rule(i) in [3,4]

    if ground:
        nal.add_background_knowledge(get_shapes_bk())

    nal.run_aba_framework(filename, id=f"shapes_r{get_rule(i)}_bk_{NUM_EXAMPLES}_ck", ground=ground)

    pos_class = shapes_metics.calculate_aba_classification_accuracy(nal,NUM_SLOTS,str(classes[0]),"c")
    neg_class = shapes_metics.calculate_aba_classification_accuracy(nal,NUM_SLOTS,str(classes[1]),"c")

    print("Positve Accuracy: ", pos_class)
    print("Negative Accuracy: ", 1- neg_class)

    return (pos_class, 1- neg_class)


def shapes_9_nal():
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

    NUM_EXAMPLES = 10
    DATA_DIR = "datasets/SHAPES_9/training_data/"
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
            ground = get_rule(i) in [3,4]
            if nal.check_prediction_quailty(prediction,THRESHOLD) and choosen_exp not in history:
                nal.populate_aba_framework(prediction, i == classes[0],include_pos=ground)
                j = j + 1
                history.append(choosen_exp)
    
    
    filename = "shapes_9_bk.aba"
    ground = get_rule(i) in [3,4]

    if ground:
        nal.add_background_knowledge(get_shapes_bk())

    nal.run_aba_framework(filename, id=f"shapes_r{get_rule(i)}_bk_{NUM_EXAMPLES}_ck", ground=ground)

    pos_class = shapes_metics.calculate_aba_classification_accuracy(nal,NUM_SLOTS,str(classes[0]),"c")
    neg_class = shapes_metics.calculate_aba_classification_accuracy(nal,NUM_SLOTS,str(classes[1]),"c")

    print("Positve Accuracy: ", pos_class)
    print("Negative Accuracy: ", 1- neg_class)

    return (pos_class, 1- neg_class)




def calcuate_ari_resutls():
    ## Setting Up Slot Attention Model
    object_info = {"object" : ["","Triangle","Circle","Square"],
                   "colour": ["","Red","Green","Blue"],
                   "size:":  ["","Large","Small"]}
    
    ckpt = torch.load(os.getcwd() + "/models/shapes_m1.pt",map_location='cpu')
    num_slots = 10

    model = SLOTS_9(
                in_shape=(3,64,64),
                width=32,
                num_slots=num_slots,
                slot_dim=32,
                routing_iters=3,
            )

    model.load_state_dict(ckpt['model_state_dict'],)
    data = "datasets/SHAPES_9/training_data/c1_s1/c1_1.png"
    label = "datasets/SHAPES_9/training_data/c1_s1/labels.txt"

    nal = NAL(model,object_info)

    ## Retrieving Attenion Map

    _, mask, = nal.run_slot_attention_model(data,num_slots)
    mask = mask.squeeze(1).numpy()

    image = Image.open(data).resize((64, 64))
    image_array = np.array(image)

    ## Calcualating ARI

    pixel_assignments = e.assign_pixels_to_clusters(image_array,mask)
    true_pixel_assignments = get_ground_truth_map(label,9).flatten()
    

    ari_score = e.adjusted_rand_score(true_pixel_assignments,pixel_assignments)
    print("ARI Score: ", ari_score)

    clustering_map = np.array(pixel_assignments).reshape((64,64))

    ## Visualaing maps 

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    clustering_map = np.array(pixel_assignments).reshape((64,64))
    true_map = true_pixel_assignments.reshape((64, 64))

    axes[0].imshow(clustering_map, cmap='viridis')
    axes[0].set_title('Clustering Map')
    axes[0].set_xlabel('Width')  
    axes[0].set_ylabel('Height')  
    plt.colorbar(ax=axes[0])

    axes[1].imshow(true_map, cmap='viridis')
    axes[1].set_title('A Map')
    axes[1].set_xlabel('Width')  
    axes[1].set_ylabel('Height')
    plt.colorbar(ax=axes[1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    shapes_4_nal()
