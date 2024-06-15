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
    postive_class = [0]
    negative_class = [1]

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


def clevr_nal_inference(img_path: str, aba_path: list[str], include_pos = False):

    """
        CLEVR Hans Classification Task

        img_path: images to perform inference on
        aba_path: list of lenght 2. Inference for CLEVR-hans is perfomed in two passes using both learnt rules
        include_pos: Boolean denoting whether to add co-oodrinates to the framework 

        Task:
            Idenfication of concept predicate c(...) denotes positive instance of class 

    
    """

    nal = get_clevr_nal_model()
    NUM_SLOTS = 11

    prediction, _ = nal.run_slot_attention_model(img_path,NUM_SLOTS,num_coords=3,isPath=False)
    nal.populate_aba_framework_inference(prediction,include_pos)

    # First Pass: Class 3 
    nal.load_framework(aba_path[0])
    all_models = nal.run_learnt_framework()
    total_model = len(all_models)

    has_predicate = check_predicate_presence(all_models,total_model)
    
    if has_predicate:
        return 2
    
    # Second Pass: Class 2
    nal.load_framework(aba_path[1])
    all_models = nal.run_learnt_framework()
    total_model = len(all_models)
   
    has_predicate = check_predicate_presence(all_models,total_model)

    if has_predicate:
        return 0
    else:
        return 1
        

if __name__ == "__main__":  
    train_nal_clevr(num_examples=20)
    


