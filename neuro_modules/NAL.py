from typing import Callable, Optional, Tuple
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

from NAL_Shapes.slots import SlotAutoencoder
import utils


class NAL:
    def __init__(self, model, object_info) -> None:

        self.data_dir = "datasets/MOCK/training_data/"
        self.object_info = object_info
        self.model = model

        self.aba_framework = ABAFramework()
        self.obj_id = 1
        self.img_id = 1


    def preprocess_image(self,image_path):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.PILToTensor(),
        ])
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image)
        
        input_batch = input_tensor.unsqueeze(0)

        input_batch = (input_batch - 127.5) / 127.5
        return input_batch

    
    def run_slot_attention_model(self,image, num_of_slots):
        self.model.eval()
        image = self.preprocess_image(image)

        with torch.no_grad():
            _, _, masks, _ , output = self.model(image)

        masks = masks.squeeze(0)
        out = output.squeeze(0)

        results =  []

        for i in range(num_of_slots):
            slot_results = []
            region = utils.get_attended_bounding_box(masks[i][0].numpy())
            slot_results.append({"attention_map": region})

            num_classes = len(list(self.object_info.keys()))
            class_info = list(self.object_info.values())
            class_name = list(self.object_info.keys())
            idx = 0
            for j in range(num_classes):
                item = class_info[j]
                end = idx + len(item)

                attribute = torch.argmax(out[i][idx:end]).item()
                attribute_confidence = out[i][idx:end][attribute].item()
                
                pred = {class_name[j] : item[attribute],
                        "confidence": attribute_confidence}
                
                slot_results.append(pred)
                idx += len(item)

            results.append(slot_results)
        return results
    

    def check_prediction_quailty(self,predictions,threshold):
        min_confidence = 100

        for slot in predictions:
            total_confidence = 1
            slot = slot[1:]

            for item in slot:
                total_confidence = total_confidence * item["confidence"]

            if total_confidence < min_confidence:
                min_confidence = total_confidence
            
        return min_confidence > threshold
    

    def is_valid(self,slot):
        for item in slot:
            if "type" in list(item.keys()):
                return item["type"] == ""

        return False
    
    
    def add_background_knowledge(self, rules : list[str]):
        for rule in rules:
            self.aba_framework.add_bk_rule(rule)
    
    def populate_aba_framework(self, slots: list[dict], isPositive: bool, include_pos = False):
    
        img_label = f"img_{self.img_id}"

        attributes = list(self.object_info.keys())

        for slot in slots:

            slot = {key: value for dictionary in slot for key, value in dictionary.items()}

            if slot["object"] == "":
                continue

            obj_label = f"object_{self.obj_id}"
            self.aba_framework.add_bk_fact(img_label,pred_name="in",arity=2, args=[img_label,obj_label])

            for att in attributes:
                pred = slot[att].lower()
                
                if pred == "":
                    continue
                self.aba_framework.add_bk_fact(img_label,pred_name=pred,arity=1, args=[obj_label])

            
            pos_info = slot["attention_map"]

            if (include_pos and pos_info[0] != -1 ):
                slot_info = [img_label, obj_label] + [str(c) for c in list(pos_info)]
                self.aba_framework.add_bk_fact(img_label,pred_name="box",arity=6,args=slot_info)

            self.obj_id += 1

        self.aba_framework.add_bk_fact(img_label,pred_name="image",arity=1, args=[img_label])
        self.aba_framework.add_example(pred="c", args=[img_label], isPositive=isPositive)
        self.img_id += 1

    
    def populate_aba_framework_inference(self,slots: list[dict], include_pos = False):
        img_label = f"img_{self.img_id}"

        attributes = list(self.object_info.keys())

        for slot in slots:

            slot = {key: value for dictionary in slot for key, value in dictionary.items()}

            if slot["object"] == "":
                continue

            obj_label = f"object_{self.obj_id}"
            self.aba_framework.add_inference_bk_fact(pred_name="in",arity=2, args=[img_label,obj_label])

            for att in attributes:
                pred = slot[att].lower()
                
                if pred == "":
                    continue
                self.aba_framework.add_inference_bk_fact(pred_name=pred,arity=1, args=[obj_label])

            
            pos_info = slot["attention_map"]

            if (include_pos and pos_info[0] != -1 ):
                slot_info = [img_label, obj_label] + [str(c) for c in list(pos_info)]
                self.aba_framework.add_inference_bk_fact(pred_name="box",arity=6,args=slot_info)

            self.obj_id += 1

        self.aba_framework.add_inference_bk_fact(pred_name="image",arity=1, args=[img_label])
        self.img_id += 1
        

    def run_aba_framework(self, filename="aba_framework.aba", id=0, ground=False):
        self.aba_framework.write_aba_framework(filename)

        if ground:
            self.aba_framework.ground_aba_framework(filename)
        self.aba_framework.set_aba_sovler_path("symbolic_modules/aba_asp/aba_asp.pl")
        self.aba_framework.run_aba_framework(id)



    def get_classificaiton_result(self, s_models, c_label) -> bool: 
        
        present = 0
        total_model = len(s_models)

        for model in s_models:
            for symbol in model:
                if f"{c_label}(img_"in str(symbol): 
                    present+=1

        
        absent = total_model - present
        return present > absent
