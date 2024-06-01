import os
import random
import torch

from datasets.SHAPES_4.SHAPES4 import SHAPESDATASET as SHAPESDATASET_4
from datasets.SHAPES_9.SHAPES import SHAPESDATASET as SHAPESDATASET_9

from neuro_modules.slots import SlotAutoencoder
from neuro_modules.NAL import NAL


def get_class_dict(object_dict : dict):
    length_dict = {}

    for key, value in object_dict.items():
        length_dict[key] = len(value)

    return length_dict

def get_shape_9_nal_model():
    object_info = {"object" : ["Triangle","Circle","Square"],
                   "colour": ["Red","Green","Blue"],
                   "size:":  ["Large","Small"]}

    ckpt = torch.load(os.getcwd() + "/models/shapes_m1.pt",map_location='cpu')

    model = SlotAutoencoder(
            in_shape=(3,64,64),
            width=32,
            num_slots=10,
            slot_dim=32,
            routing_iters=3,
            classes= get_class_dict(object_info)
        )

    model.load_state_dict(ckpt['model_state_dict'],)

    nal = NAL(model,object_info)

    return nal


def add_shape_bk(nal):

    bk_rules = [("above","above(S1,S2,I) :- position(I,S1,X1,Y1), position(I,S2,X2,Y2), Y1 - Y2 > 0."),
                ("right","right(S1,S2,I) :- position(I,S1,X1,Y1), position(I,S2,X2,Y2), X1 - X2 > 0.")]
    
    for pred, rule in bk_rules:
        nal.add_background_knowledge(rule,pred)
        
    return True

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
    if num_class <= 12:
        return 6

def shapes_9_nal_training(num_examples: int, class_ids: list):
    
    nal = get_shape_9_nal_model()

    NUM_SLOTS = 10
    THRESHOLD = 0.9

    data = SHAPESDATASET_9(cache=False)
    data = data.get_SHAPES_dataset()

    total_items = len(data[class_ids[0]])
    ## Populating ABA Framework with Examples

    print("Finding Examples....")

    for i in class_ids:
        j = 0
        history = []
        while j < num_examples:
            choosen_exp = random.randint(0, total_items - 2)
            img_path = data[i][choosen_exp]  ## Tuple of (img_path, label_path)
            prediction, _ = nal.run_slot_attention_model(img_path[0],NUM_SLOTS)
            ground = get_rule(i) in [4,5]  
            if nal.check_prediction_quailty(prediction,THRESHOLD) and choosen_exp not in history:
                nal.populate_aba_framework(prediction, i == class_ids[0],include_pos=ground)
                j = j + 1
                history.append(choosen_exp)            

    
    filename = f"shapes_9_bk_r{get_rule(i)}.aba"

    if ground:
        add_shape_bk(nal)

    print("Running Framework")

    pred_order = ["square", "circle", "triangle", "red", "green", "blue", "small", "large", "above", "left"]

    nal.run_aba_framework(filename, id= get_rule(i), ground= ground, order= pred_order)


def shapes_9_nal_training_general(num_examples: int, class_ids: list):
    
    nal = get_shape_9_nal_model()

    NUM_SLOTS = 10
    THRESHOLD = 0.9

    data = SHAPESDATASET_9(cache=False)
    data = data.get_SHAPES_dataset()

    total_items = len(data[class_ids[0]])
    ## Populating ABA Framework with Examples

    print("Finding Examples....")

    for i in class_ids:
        j = 0
        history = []
        while j < num_examples:
            choosen_exp = random.randint(0, total_items - 2)
            img_path = data[i][choosen_exp]  ## Tuple of (img_path, label_path)
            labels = nal.run_slot_attention_kmean(img_path[0])
            nal.populate_aba_framework_general(labels, i == class_ids[0])
            j = j + 1
            history.append(choosen_exp)            

    
    filename = f"shapes_9_bk_r{get_rule(i)}.aba"

    print("Running Framework....")

    nal.run_aba_framework(filename, id= get_rule(i))

def shapes_9_nal_inference(img_path: str, aba_path: str, include_pos = False):
    nal = get_shape_9_nal_model()

    NUM_SLOTS = 10

    prediction, _ = nal.run_slot_attention_model(img_path,NUM_SLOTS)
    nal.populate_aba_framework_inference(prediction,include_pos)

    """
        Classification Task
        if predicate c(...) is in majority stable model we classify image as positve 
    
    """

    # img = f":- c(img_{nal.img_id})."
    all_models = nal.run_learnt_framework(aba_path)
    # r_models = nal.run_learnt_framework(aba_path,img)

    total_model = len(all_models)
    present = 0


    for model in all_models:
        for symbol in model:
            if f"c(img_"in str(symbol): 
                present+=1

        
        absent = total_model - present
        return present > absent
    
    return 0


if __name__ == "__main__":

    ## Example Training and Inference
    
    # shapes_4_nal_training(10,[1,2])
    # test_img = "/mnt/d/fyp/SHAPES_9/training_data/c1_s10/c1_10.png"
    # aba_path = "shapes_9_bk_r1_SOLVED.aba"
    # prediction = shapes_9_nal_inference(test_img,aba_path)

    # if prediction:
    #     print("positive")
    # else:
    #     print("negative")


    classes = [[7,8]]
    for c in classes:
        shapes_9_nal_training(10,c)

    # test_img = "/mnt/d/fyp/SHAPES_9/training_data/c1_s10/c1_10.png"
    # test_img2 = "/mnt/d/fyp/SHAPES_9/training_data/c2_s10/c2_10.png"

    # nal = get_shape_9_nal_model()

    # labels = nal.run_slot_attention_kmean(test_img)
    # nal.populate_aba_framework_general(labels,True)
    # labels = nal.run_slot_attention_kmean(test_img2)
    # nal.populate_aba_framework_general(labels,False)
    # nal.run_aba_framework()


