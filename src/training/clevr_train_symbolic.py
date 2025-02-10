import os
import torch
import matplotlib.pyplot as plt

from datasets.CLEVR.CLEVR import CLEVRHans
from neuro_modules.NAL import NAL
from neuro_modules.slots import SlotAutoencoder


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

    nal = NAL(model,None,object_info)

    return nal

def check_predicate_presence(s_models, total_model, pred_name="c"):
    present = 0

    for model in s_models:
        for symbol in model:
            if f"{pred_name}(img_"in str(symbol): 
                present+=1
                break

        
    absent = total_model - present
    
    return present > absent


def train_nal_clevr(num_examples: int):

    NUM_SLOTS = 11  # Number of slots in the slot model 
    THRESHOLD = 0.7 # Theshold for confidence of predicitons 

    data = CLEVRHans(transform=CLEVRHans.get_transform())

    clevr_NAL = get_clevr_nal_model()
    clevr_NAL.dataset = data
    postive_class = [0]  # Positve Class [0,1,2]
    negative_class = [1] # Negative Class [0,1,2]

    print("Choosing Examples...")

    
    p_examples , n_examples = clevr_NAL.choose_example(postive_class, negative_class, num_examples)


    for eg in p_examples:
        data_point = data[eg]
        prediction, _ = clevr_NAL.run_slot_attention_model(data_point["input"],NUM_SLOTS,num_coords=3,isPath=False)
        if clevr_NAL.check_prediction_quailty(prediction,THRESHOLD):
            clevr_NAL.populate_aba_framework(prediction,True)


    for eg in n_examples:
        data_point = data[eg]
        prediction, _ = clevr_NAL.run_slot_attention_model(data_point["input"],NUM_SLOTS,num_coords=3,isPath=False)
        if clevr_NAL.check_prediction_quailty(prediction,THRESHOLD):
            clevr_NAL.populate_aba_framework(prediction,False)

    
    filename = f"clevr_bk_c6.aba"

    print("Running ABA Learning Algorithm...")

    clevr_NAL.run_aba_framework(filename)






