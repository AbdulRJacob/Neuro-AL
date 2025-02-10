import os
import yaml
import torch

from data.SHAPES import SHAPESDATASET
from models.slot_ae import SlotAutoencoder
from models.NAL import NAL


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

    nal = NAL(model,None,object_info)

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

def shapes_nal_training(num_examples: int, class_ids: list, order = False):
    
    NUM_SLOTS = 10
    THRESHOLD = 0.7

    data = SHAPESDATASET(cache=True,transform=SHAPESDATASET.get_transform())

    shapes_NAL = get_shape_9_nal_model()
    shapes_NAL.dataset = data
    postive_class = class_ids[0]
    negative_class = class_ids[1]
    ground = get_rule(postive_class) in [4,5]
    

    print("Choosing Examples...")

    p_examples , n_examples = shapes_NAL.choose_example([postive_class], [negative_class], num_examples)


    for eg in p_examples:
        data_point = data[eg]
        prediction, _ = shapes_NAL.run_slot_attention_model(data_point["input"],NUM_SLOTS,num_coords=2,isPath=False)
        if shapes_NAL.check_prediction_quailty(prediction,THRESHOLD):
            shapes_NAL.populate_aba_framework(prediction,True,include_pos=ground)

    for eg in n_examples:
        data_point = data[eg]
        prediction, _ = shapes_NAL.run_slot_attention_model(data_point["input"],NUM_SLOTS,num_coords=2,isPath=False)
        if shapes_NAL.check_prediction_quailty(prediction,THRESHOLD):
            shapes_NAL.populate_aba_framework(prediction,False,include_pos=ground)
    
    filename = f"shapes_9_bk_r{get_rule(postive_class)}.aba"

    if ground:
        add_shape_bk(shapes_NAL)

    print("Running Framework")

    if order:
        pred_order = ["square", "circle", "triangle", "red", "green", "blue", "small", "large", "above", "left"]
        shapes_NAL.run_aba_framework(filename, id=get_rule(postive_class), ground= ground, order=pred_order)
    else:
        shapes_NAL.run_aba_framework(filename, id=get_rule(postive_class), ground= ground)

if __name__ == "__main__":

    ## Example Training and Inference

    shapes_nal_training(num_examples=10,class_ids=[1,2]) 

    