import numpy as np
import torch
import pickle
from collections import Counter
import heapq
from PIL import Image
from torchvision import transforms
from symbolic_modules.aba_framework import ABAFramework
import neuro_modules.utils as utils

class NAL:
    def __init__(self, model, dataset, object_info) -> None:

        self.dataset= dataset
        self.object_info = object_info
        self.model = model

        self.aba_framework = ABAFramework()
        self.obj_id = 1
        self.img_id = 1


    def preprocess_image(self, image , isPath=True):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.PILToTensor(),
        ])

        if isPath:
            image = Image.open(image).convert('RGB')

        input_tensor = transform(image)
        
        input_batch = input_tensor.unsqueeze(0)

        input_batch = (input_batch - 127.5) / 127.5
        return input_batch
    

    def run_slot_attention_kmean(self, image):
        self.model.eval()
        image = self.preprocess_image(image)

        with torch.no_grad():
            _, _, _, slots , _ = self.model(image)

        with open("models/kmeans_model.pkl", "rb") as f:
            kmeans = pickle.load(f)

        flattened_slots = slots.view(-1, slots.size(-1))
        flattened_slots_np = flattened_slots.detach().numpy()

        labels = kmeans.predict(flattened_slots_np)

        return labels


    def run_slot_attention_model(self, image, num_of_slots, num_coords = 2, isPath=True):
        self.model.eval()
        image = self.preprocess_image(image, isPath)

        with torch.no_grad():
            _, _, masks, _ , output = self.model(image)

        masks = masks.squeeze(0)
        out = output.squeeze(0)

        coords = (out[:, :num_coords] * 200).to(torch.int32)

        real = out[:, -1]
        out = out[:, num_coords:]
        

        results =  []

        for i in range(num_of_slots):
            slot_results = []
            slot_results.append({"object_position": coords[i].tolist()})
            slot_results.append({"real": real[i].tolist()})
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
        
        return results, masks
    

    def check_prediction_quailty(self,predictions,threshold):
        min_confidence = 100

        for slot in predictions:
            total_confidence = 1
            
            if slot[1]['real'] < 0.5:
                continue

            slot = slot[2:]

            for item in slot:
                total_confidence = total_confidence * item["confidence"]

            if total_confidence < min_confidence:
                min_confidence = total_confidence
            
        return min_confidence > threshold
    

    def choose_example(self, p_eg: list[int], n_eg: list[int], num_examples):
        data = self.dataset.get_all_data()
        rep_postive = []
        rep_negative = []
        for (idx, info) in data:
            _, _, _, slots, _ = self.model(info["input"].unsqueeze(0).float())
            combined_slots = utils.aggregate_slots(slots.detach().numpy())

            if np.argmax(info["class"]) in p_eg:
                rep_postive.append((idx, combined_slots))
            if np.argmax(info["class"]) in n_eg:
                rep_negative.append((idx, combined_slots))

        print("Finding Samples...")
        diverse_samples_pos = utils.get_diverse_slots(rep_postive,num_examples)
        diverse_samples_neg = utils.get_diverse_slots(rep_negative,num_examples)

        return diverse_samples_pos, diverse_samples_neg

    
    def add_background_knowledge(self, rule : str, pred_name: str):
            self.aba_framework.add_bk_rule(rule,pred_name=pred_name)


    def populate_aba_framework_general(self, properties, isPositive, concept="c"):
        img_label = f"img_{self.img_id}"

        frequency = Counter(properties)

        # Get the k most frequent numbers
        important_properties = heapq.nlargest(3, frequency.keys(), key=frequency.get)

        for prop in important_properties:
            self.aba_framework.add_bk_fact(img_label,pred_name=f"attr_{prop}",arity=1, args=[img_label])

        
        self.aba_framework.add_bk_fact(img_label,pred_name="image",arity=1, args=[img_label])

        self.aba_framework.add_example(pred=concept, args=[img_label], isPositive=isPositive)
        self.img_id += 1

    
    def populate_aba_framework(self, slots: list[dict], isPositive: bool, include_pos = False, concept="c"):
    
        img_label = f"img_{self.img_id}"

        attributes = list(self.object_info.keys())

        for slot in slots:

            slot = {key: value for dictionary in slot for key, value in dictionary.items()}

            if slot["real"] < 0.5:
                continue
         
            obj_label = f"object_{self.obj_id}"
            self.aba_framework.add_bk_fact(img_label,pred_name="in",arity=2, args=[img_label,obj_label])

            for att in attributes:
                pred = slot[att].lower()
                
                if pred == "":
                    continue
                self.aba_framework.add_bk_fact(img_label,pred_name=pred,arity=1, args=[obj_label])

            
            pos_info = slot["object_position"]

            if (include_pos):
                slot_info = [img_label, obj_label] + [str(c) for c in list(pos_info)]
                self.aba_framework.add_bk_fact(img_label,pred_name="position",arity=4,args=slot_info)

            self.obj_id += 1

        self.aba_framework.add_bk_fact(img_label,pred_name="image",arity=1, args=[img_label])
        self.aba_framework.add_example(pred=concept, args=[img_label], isPositive=isPositive)
        self.img_id += 1

    
    def populate_aba_framework_inference(self,slots: list[dict], include_pos = False):
        img_label = f"img_{self.img_id}"

        attributes = list(self.object_info.keys())

        for slot in slots:

            slot = {key: value for dictionary in slot for key, value in dictionary.items()}

            if slot["real"] < 0.5:
                continue

            obj_label = f"object_{self.obj_id}"
            self.aba_framework.add_inference_bk_fact(pred_name="in",arity=2, args=[img_label,obj_label])

            for att in attributes:
                pred = slot[att].lower()
                
                if pred == "":
                    continue
                self.aba_framework.add_inference_bk_fact(pred_name=pred,arity=1, args=[obj_label])

            
            pos_info = slot["object_position"]

            if (include_pos):
                slot_info = [img_label, obj_label] + [str(c) for c in list(pos_info)]
                self.aba_framework.add_inference_bk_fact(pred_name="position",arity=4,args=slot_info)


            self.obj_id += 1

        self.aba_framework.add_inference_bk_fact(pred_name="image",arity=1, args=[img_label])
        self.img_id += 1
        

    def run_aba_framework(self, filename="aba_framework.aba", id=0, ground=False ,order=[]):
        self.aba_framework.write_aba_framework(filename)

        if ground:
            self.aba_framework.ground_aba_framework(filename)

        if order != []:
            order.insert(0,"image")
            order.append("in")

            self.aba_framework.write_ordered_facts(order)
        
        self.aba_framework.set_aba_sovler_path("symbolic_modules/aba_asp/aba_asp.pl")
        self.aba_framework.run_aba_framework(id)

    def load_framework(self,filepath):
        self.aba_framework.load_solved_framework(filepath)

    def run_learnt_framework(self, restrict=""):
        models = self.aba_framework.compute_stable_models(restrict)

        return models

