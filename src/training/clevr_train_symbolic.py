import os
import torch
import yaml
import logging
import matplotlib.pyplot as plt

from data.CLEVR import CLEVRHans
from models.NAL import NAL
from models.slot_ae import SlotAutoencoder


def get_class_dict(object_dict : dict):
    length_dict = {}

    for key, value in object_dict.items():
        length_dict[key] = len(value)

    return length_dict

def get_config():
    with open("config/clevr_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    return config

config = get_config()



def get_clevr_nal_model():
    object_info = {"object" : ["cube", "sphere", "cylinder"],
                   "colour": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
                   "size": ["large","small"],
                   "material" : ["rubber","metal"]
                   }

    model_dir = config['sym_training']['model_path']

    ckpt = torch.load(model_dir,map_location='cpu')

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


def train_nal_clevr(num_examples: int, train_order: list[tuple[int]]):

    NUM_SLOTS = 11  # Number of slots in the slot model 
    THRESHOLD = 0.2 # Theshold for confidence of predicitons

    logging.info("Loading Data")
    data_dir = config["data"]["data_dir_clvr_hans"]

    data = CLEVRHans(transform=CLEVRHans.get_transform(),data_dir=data_dir)

    clevr_NAL = get_clevr_nal_model()
    clevr_NAL.dataset = data

    logging.info("Entering Example Loop")

    for i in range(len(train_order)):
        postive_class = [train_order[i][0]]
        negative_class = [train_order[i][1]]

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

        
        filename = config['training']['model_dir'] + f"clevr_bk_pos_class_{postive_class[0] + 1}.aba"

        print("Running ABA Learning Algorithm...")
        logging.info("Running ABA Learning Algorithm...")

        clevr_NAL.run_aba_framework(filename)



if __name__ == "__main__":


    num_examples = config['sym_training']['num_examples']
    class_order = config['sym_training']['class_order']


    train_order = tuples = [(class_order[i], class_order[i+1]) for i in range(len(class_order) - 1)]


    train_nal_clevr(num_examples,train_order) 





