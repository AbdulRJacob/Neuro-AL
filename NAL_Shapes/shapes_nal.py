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

from slots import SlotAutoencoder
from data_structures.aba_framework import ABAFramework


class SHAPES_NAL:
    def __init__(self) -> None:

        ## Dataset
        self.dataset_path = "datasets/MOCK/training_data/"
        self.SHAPES_CLASSESS = 20
        self.TRAINING_SET_SIZE = 200

        ## Neuro Vars
        self.neuro_model = None
        self.model_params: str = ""
        self.num_of_slots = 10

        ## Symoblic Vars
        self.img_id = 1
        self.obj_id = 1
        self.ruleId = 1
        self.posId = 1
        self.negId = 1

        ## ABA Framework
        self.aba_framework = ABAFramework()
        self.size_labels = {"L": "large", "S": "small", '' : "large"}
        self.slot_labels = {0: "X1=0, Y1=0, X2=95, Y2=95.", 
                            1: "X1=100, Y1=0, X2=195, Y2=95.",
                            2: "X1=200, Y1=0, X2=295, Y2=95.", 
                            3: "X1=0, Y1=100, X2=95, Y2=195.", 
                            4: "X1=100, Y1=100, X2=195, Y2=195.", 
                            5: "X1=200, Y1=100, X2=295, Y2=195.", 
                            6: "X1=0, Y1=200, X2=95, Y2=295.", 
                            7: "X1=100, Y1=200, X2=195, Y2=295.", 
                            8: "X1=200, Y1=200, X2=295, Y2=295."}


    def init_model(self):
        ckpt = torch.load(self.model_params,map_location='cpu')

        model = SlotAutoencoder(
                in_shape=(3,64,64),
                width=32,
                num_slots=10,
                slot_dim=32,
                routing_iters=3,
            )

        model.load_state_dict(ckpt['model_state_dict'],)

        self.neuro_model = model

    def get_SHAPES_dataset(self, num_of_examples: int):

        values = []
        dataset = {}

        for i in range(1,self.SHAPES_CLASSESS+1):
            dataset[i] = []
            values = []
            for j in range(1,num_of_examples+1):
                id = random.randint(0,self.TRAINING_SET_SIZE -1)
                while id in values:
                    id = random.randint(0,self.TRAINING_SET_SIZE -1)

                values.append(id)
                img_path = self.dataset_path + f"c{i}_s{id}/c{i}_{id}.png"
                label = self.dataset_path + f"c{i}_s{id}/labels.txt"
                dataset[i].append((img_path,label))


        return dataset

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


    def to_numpy(self, x):
        return x.cpu().detach().numpy()

    def renormalize(self, x):
        # x = x.clamp(min=-1, max=1)
        return x / 2. + 0.5  # [-1, 1] to [0, 1]

    @torch.no_grad()
    def visualise_slots(self,img_path, idx=0):
        img_path = self.preprocess_image(img_path)
        recon_combined, recons, masks, slots, _  = self.neuro_model(img_path)
        image = self.renormalize(img_path)[idx]
        recon_combined = self.renormalize(recon_combined)[idx]
        recons = self.renormalize(recons)[idx]
        masks = masks[idx]
        
        image = self.to_numpy(image.permute(1,2,0))
        recon_combined = self.to_numpy(recon_combined.permute(1,2,0))
        recons = self.to_numpy(recons.permute(0,2,3,1))
        masks = self.to_numpy(masks.permute(0,2,3,1))
        slots = self.to_numpy(slots)

        num_slots = 10
        fig, ax = plt.subplots(1, num_slots + 2, figsize=(18, 4))
        ax[0].imshow(image)
        ax[0].set_title('Input')
        ax[1].imshow(recon_combined)
        ax[1].set_title('Reconstruction')
        for i in range(num_slots):
            ax[i + 2].imshow(recons[i] * masks[i] + (1 - masks[i]))
            # ax[i + 2].imshow(recons[i] * masks[i])
            # ax[i + 2].imshow(masks[i])
            ax[i + 2].set_title('Slot %s' % str(i + 1))
        for i in range(len(ax)):
            ax[i].grid(False)
            ax[i].axis('off')

        plt.savefig('slot_vis.png')
        # plt.tight_layout()


    def get_attended_bounding_box(self,mask):
        # Threshold the mask to identify the attended region
        threshold_value = 0.1
        
        attended_region = (mask > threshold_value).astype(int)

        # Find the indices of non-zero elements in the attended region
        non_zero_indices = np.nonzero(attended_region)

        # Calculate the bounding box coordinates
        x_min = np.min(non_zero_indices[1])
        y_min = np.min(non_zero_indices[0])
        x_max = np.max(non_zero_indices[1])
        y_max = np.max(non_zero_indices[0])

        # Represent the bounding box coordinates
        bounding_box = (x_min, y_min, x_max, y_max)

        return bounding_box


    def get_labels(self, shape: int ,colour : int, size: int) -> Tuple[str,str,str]:
        shape_labels = ["","Triangle","Circle","Square"]
        colour_labels = ["","Red","Green","Blue"]
        size_labels = ["","L","S"]

        return shape_labels[shape], colour_labels[colour], size_labels[size]
    
    def run_model(self,image):
        self.neuro_model.eval()

        with torch.no_grad():
            _, _, masks, _ , output = self.neuro_model(image)

        masks = masks.squeeze(0)
        out = output.squeeze(0)
        res =  []

        for i in range(10):
            region = self.get_attended_bounding_box(masks[i][0].numpy())
            # region = (0,0,0,0)
            shape = torch.argmax(out[i][:4]).item()
            shape_conf = out[i][:4][shape].item()
            colour = torch.argmax(out[i][4:8]).item()
            colour_conf = out[i][4:8][colour].item()
            size = torch.argmax(out[i][8:]).item()
            size_conf = out[i][8:][size].item()

            res.append([(region,1),(shape,shape_conf),(colour,colour_conf),(size,size_conf)])
        
        return res


    def get_prediction(self, img_path: str):
        image  = self.preprocess_image(img_path)
        results = self.run_model(image)

        set_results = []
        image_conf = 0


        for info in results:
            pos, pc = info[0]
            shape, sc = info[1]
            colour, cc = info[2]
            size, zc = info[3]

            shape, colour, size = self.get_labels(shape,colour,size)

            total_conf = pc + sc + cc + zc
            image_conf += total_conf

            set_results.append(((pos,shape,colour,size), total_conf))


        return set_results, image_conf / 9
    
    def get_ground_truth(self, label_path:str ):
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
    

    def init_aba_framework(self, rules : list[str]):
        for rule in rules:
            self.aba_framework.add_bk_rule(rule)
    

    def populate_aba_framework(self,entities: set[Tuple[Tuple[Tuple[int,int,int,int],str,str,str],float]] ,isPositive: bool):
    
        img_label = f"img_{self.img_id}"

        for entity in entities:
            pos, shape, colour, size = entity[0]
            
            if shape == "":
                continue

            shape = shape.lower()
            shape_label = f"{shape}_{self.obj_id}"

            self.aba_framework.add_bk_fact(img_label,pred_name="in",arity=2, args=[img_label,shape_label])
            self.aba_framework.add_bk_fact(img_label,pred_name=shape,arity=1, args=[shape_label])

            if not colour == "":
                colour = colour.lower()
                self.aba_framework.add_bk_fact(img_label,pred_name=colour,arity=1, args=[shape_label])


            size = self.size_labels[size]
            self.aba_framework.add_bk_fact(img_label,pred_name=size,arity=1, args=[shape_label])
            slot_info = [img_label, shape_label] + [str(c) for c in list(pos)]
            self.aba_framework.add_bk_fact(img_label,pred_name="box",arity=6,args=slot_info)
            self.obj_id += 1

        
        self.aba_framework.add_bk_fact(img_label,pred_name="image",arity=1, args=[img_label])
        self.aba_framework.add_example(pred="c", args=[img_label], isPositive=isPositive)
        
        self.img_id += 1

    def populate_aba_framework_inference(self,entitiies):
        keys = list(entitiies.keys())
        values = list(entitiies.values())

        img_label = f"img_{self.img_id}"

        for i in range(len(keys)):
            slot = int(keys[i])
            vals = values[i][0]
            shape, colour, size = vals

            if shape == "":
                continue

            shape = shape.lower()
            colour = colour.lower()

            if colour == "":
                colour = "blue"

            size = self.size_labels[size]
            shape_label = f"{shape}_{self.obj_id}"

            self.aba_framework.add_inference_bk_fact(pred_name="in",arity=2, args=[img_label,shape_label])
            self.aba_framework.add_inference_bk_fact(pred_name=shape,arity=1, args=[shape_label])
            self.aba_framework.add_inference_bk_fact(pred_name=colour,arity=1, args=[shape_label])
            self.aba_framework.add_inference_bk_fact(pred_name=size,arity=1, args=[shape_label])
            # slot_rule = f"box(I,S,X1,Y1,X2,Y2) :- I={img_label}, S={shape_label},{self.slot_labels[slot]}"
            # self.aba_framework.add_bk_rule(slot_rule,img_label)
            self.obj_id += 1

        
        self.aba_framework.add_inference_bk_fact(pred_name="image",arity=1, args=[img_label])
        
        self.img_id += 1

    def get_classificaiton_result(self, s_models, c_label) -> bool: 
        is_instance = False

        for symbol_sequence in s_models:
            for symbol in symbol_sequence:
                if f"{c_label}(img_"in str(symbol): 
                    is_instance = True

        return is_instance


if __name__ == "__main__":  

    shapes_nal = SHAPES_NAL()
    
    # ## Initialisnig Model

    model = SlotAutoencoder(in_shape=(3,64,64), width=32, num_slots=10, slot_dim=32, routing_iters=3)
    shapes_nal.model_params = os.getcwd() + "/models/232128_ckpt.pt"
    shapes_nal.neuro_model = model
    shapes_nal.init_model()

    shapes_nal.visualise_slots("sample_s0/sample_0.png")
    exit()


    ## Populating ABA Framework

    NUM_EXAMPLES = 5
    classes = [1,2]

    data = shapes_nal.get_SHAPES_dataset(NUM_EXAMPLES)

    bk_rules = ["above(S1,S2,I) :- box(I,S1,X1,Y1,X2,Y2), box(I,S2,X12, Y12, X22,Y22), Y12 - Y2 > 0.",
                "below(S1,S2,I) :- box(I,S1,X1,Y1,X2,Y2), box(I,S2,X12, Y12, X22,Y22), Y1 - Y22 > 0.",
                "left(S1,S2,I) :- box(I,S1,X1,Y1,X2,Y2), box(I,S2,X12, Y12, X22,Y22), X2 - X12 < 0.",
                "right(S1,S2,I) :- box(I,S1,X1,Y1,X2,Y2), box(I,S2,X12, Y12, X22,Y22), X22 - X1 < 0."]
    
    shapes_nal.init_aba_framework(bk_rules)

    for i in classes : ## range(1, SHAPES_CLASSESS + 1)
        for j in range(NUM_EXAMPLES):
            img_path = data[i][j]  ## Tuple of (img_path, label_path)
            prediction, confidence = shapes_nal.get_prediction(img_path[0])
            if confidence < 2.5:
                continue
            
            shapes_nal.populate_aba_framework(prediction, i == classes[0])

    aba = shapes_nal.aba_framework

    ## Training ABA Framework 
    filename = "test_bk.aba"

    aba.write_aba_framework(filename)
    aba.ground_aba_framework(filename)
    # aba.set_aba_sovler_path("symbolic_modules/aba_asp/aba_asp.pl")

    # aba.run_aba_framework()

    ## Inference Example

    # aba.load_background_knowledge("/home/abdul/Imperial_College/Year_4/70011_Individual_Project/Neuro-AL/shapes_bk.aba")
    # aba.load_learnt_rules("/home/abdul/Imperial_College/Year_4/70011_Individual_Project/Neuro-AL/shapes_bk.sol.asp")
    # aba.load_assumptions_and_contraries("/home/abdul/Imperial_College/Year_4/70011_Individual_Project/Neuro-AL/shapes_bk.sol.aba")

    # msg = lambda x : "is a positvie instance!" if x else "is a negative instance!"

    # img_path_1 = "datasets/MOCK/testing_data/c1_s222/c1_222.png"
    # prediction, confidence = shapes_nal.get_prediction(img_path_1)
    # shapes_nal.populate_aba_framework_inference(prediction)
    # models = aba.get_prediction()
    # is_positive_example = get_classificaiton_result(models,"c")

    # print(f"Image 1 {msg(is_positive_example)}")

    # shapes_nal.aba_framework.reset_inference()

    # img_path_2 = "datasets/MOCK/testing_data/c2_s222/c2_222.png"
    # prediction, confidence = shapes_nal.get_prediction(img_path_2)
    # shapes_nal.populate_aba_framework_inference(prediction)
    # models = aba.get_prediction()
    # is_positive_example = get_classificaiton_result(models,"c")

    
    # print(f"Image 2 {msg(is_positive_example)}")



