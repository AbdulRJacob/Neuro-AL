import copy
import os
import re
import random
from enum import Enum

import numpy as np
import cv2
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor

class SHAPES:

    def __init__(self,height,width,filepath) -> None:
        self.height = height  
        self.width = width 
        self.filepath = filepath 

        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.blue = (255, 0, 0)

        self.label = []


    def generate_shape(self, rule: str, negated: bool, label: str, batch_size: int,label_offset=0):
      
        print(f"Generating Data for rule(s): \"{rule}\"")

        parsed_rules = self.parse_rule(rule)
        
        for i in range(batch_size):
            if (i % 500 == 0):
                print(f"Generated {i}/{batch_size} images")

            if (parsed_rules == []):
                print("Invalid Rule")
                return 0
            
            self.generate_image(parsed_rules,negated,label,i + label_offset)
        
        print("Done!")
        return 1
    

    def parse_rule(self, rules: list[str]): 
        
        pos_pattern = r'\b(above|right|left|below)\(([^)]*)\)'
        
        pos = ["above","below","left","right"]
        objs = [f"O{i}" for i in range(1,10)]

        parsed_rule= [] ## (Rule_ID, Predicates, Postinal_Predicates, Exception Rule)
        obj_id = []


        for i in range(len(rules)):
    
            negation = False
            head, rule = rules[i].split("-")                 ## remove head of rule  
            rule = re.split(r',(?![^()]*\))', rule)          ## retrives list of predictes in the rule

    
            if "exception" in head:
                negation = True


            if any(any(p in pred for pred in rule) for p in pos):
                # Positional Rule 
                position  = re.findall(pos_pattern, ' '.join(rule))
                item_list = []
                for obj in objs:
                    item = list(filter(lambda i: obj in i, rule))
                    if item != []:
                        item_list.append(item)
                        obj_id.append(obj)

                parsed_rule.append((Rule.Postional, [' '.join(i) for i in item_list],position,negation))

            else:
                # Basic Rule 
                for obj in objs:
                    item = list(filter(lambda i: obj in i, rule))
                    
                    if item != []:
                        parsed_rule.append((Rule.BASIC,' '.join(item),None,negation))
                        obj_id.append(obj)

        output = []
        entry = {}

        # Initialse dic of shapes included in rules 
        for i in range(len(obj_id)):
            entry[obj_id[i]] = {"shape": "", "colour": "", "size": ""}

        for i in range(len(parsed_rule)):            

            if (parsed_rule[i][0] == Rule.BASIC):
                rule = parsed_rule[i][1] 
                entry[obj_id[i]]["shape"] = self.get_shape(rule)
                entry[obj_id[i]]["colour"] = self.get_colour(rule)
                entry[obj_id[i]]["size"] = self.get_size(rule)
                output.append((parsed_rule[i][0], entry, obj_id[i], None, parsed_rule[i][-1]))
            elif (parsed_rule[i][0] == Rule.Postional):
                rules = parsed_rule[i][1]
                for j in range(len(obj_id) - 1): ## obj_id corresponds to each positional rule element  ## TODO: Bug with index taking objs from pre exceptions
                    entry[obj_id[j]]["shape"] = self.get_shape(rules[j])
                    entry[obj_id[j]]["colour"] = self.get_colour(rules[j])
                    entry[obj_id[j]]["size"] = self.get_size(rules[j])

                output.append((parsed_rule[i][0], entry, obj_id, parsed_rule[i][2],parsed_rule[i][-1]))

        return output
    

    def check_image(self,image,rule,negated,idx=0):
        
        if idx >= len(rule):
            return True

        r, r_obj, obj_id , pos, negated = rule[idx]
    
        cond_satisfied = False
        same_shape = lambda i,s: i["shape"] == s["shape"] and (i["colour"] == s["colour"] or s["colour"] == None ) and (i["size"] == s["size"] or s["size"] == None)
        
        ## Rule Basic: Checks whether the image contains the object specified
        if r == Rule.BASIC:
            r_obj = r_obj[obj_id]
            for i in image:
                cond = same_shape(i,r_obj)
                if cond:
                    cond_satisfied = True

            if negated:
                return not cond_satisfied and self.check_image(image,rule,negated,idx + 1)
            else:
                return cond_satisfied and self.check_image(image,rule,negated,idx + 1)
            
        ## Rule Postinal: 
        elif r == Rule.Postional:
            obj_pos = {}
            present = [False for i in obj_id]
            cond = False
            for i in image:
                for j in range(len(obj_id)):
                    obj = obj_id[j]
                    cond = same_shape(i,r_obj[obj])
                    if cond: 
                        obj_pos[obj] = i["quad"]
                        present[j] = True
                        cond = False
                        continue

            if not all(present):
                return negated 
            else:
                if negated:
                    return  not self.check_position(obj_pos,pos) and self.check_image(image,rule,negated,idx + 1)
                else:
                    return self.check_position(obj_pos,pos) and self.check_image(image,rule,negated,idx + 1)
            
    def check_position(self,obj_pos,postions,idx=0):
        
        if idx >= len(postions):
            return True
        
        p = postions[idx]
        cur_pos = self.get_position(p[0])
        obj_1 = p[1].split(",")[0]
        obj_2 = p[1].split(",")[1]

        if cur_pos == Position.RIGHT:
            temp = obj_1
            obj_1 = obj_2
            obj_2 = temp 
            cur_pos = Position.LEFT

        if cur_pos == Position.BELOW:
            temp = obj_1
            obj_1 = obj_2
            obj_2 = temp 
            cur_pos = Position.ABOVE

        if cur_pos == Position.LEFT:
            if obj_pos[obj_2] in [0,3,6]:
                return False
            elif obj_pos[obj_2] in [1,4,7] and obj_pos[obj_1] in [0,3,6] or obj_pos[obj_2] in [2,5,8] and obj_pos[obj_1] in [0,1,3,4,6,7]:
                return  True and self.check_position(obj_pos,postions,idx+1)
            
        if cur_pos  == Position.ABOVE:
            if obj_pos[obj_2] in [0,1,2]:
                return False 
            elif obj_pos[obj_2] in [3,4,5] and obj_pos[obj_1] in [0,1,2] or obj_pos[obj_2] in [6,7,8] and obj_pos[obj_1] in [0,1,2,3,4,5]:
                return True and self.check_position(obj_pos,postions,idx+1)
        



    def draw_shape(self, grid_info,image):

        s = ""
        sz = ""
        quad, shape, colour, size = list(grid_info.values())

        if colour == "":
            colour = self.get_colour(colour)

        if shape == Shape.TRIANGLE:
            image = self.draw_triangle(image, quad, colour,size)
            s = "Triangle"

        elif shape == Shape.SQUARE:
            image = self.draw_square(image, quad, colour,size)
            s = "Square"

        elif shape == Shape.CIRCLE:
            image = self.draw_circle(image, quad, colour,size)
            s = "Circle"

        if size == Size.LARGE:
            sz = "L"
        else:
            sz = "S"
        
        self.get_label(s,colour,quad,sz)


        return image
    
    def draw_square(self, image, quadrant, colour, size=""):
        points = {Size.LARGE: (5,5,95,95), Size.SMALL: (30,30,70,70)}

        (x_1,y_1,x_2,y_2) = points[size]

        offset_x , offset_y = self.get_offset(quadrant)

        cv2.rectangle(image, (offset_x + x_1, offset_y + y_1), (offset_x + x_2, offset_y + y_2), colour, -1) 

        return image
    
    def draw_triangle(self, image, quadrant, colour,size=""):

        points = {Size.LARGE: (5,95,95,95,50,95,90), Size.SMALL: (30,70,70,70,50,70,40)}

        (x_1,y_1,x_2,y_2,x_3,y_3,s) = points[size]
    
        offset_x , offset_y = self.get_offset(quadrant)
       
        c1 = (offset_x + x_1, offset_y + y_1)
        c2 = (offset_x + x_2, offset_y + y_2)

        h = int((3**0.5 / 2) * s) 
        c3 = (offset_x + x_3, offset_y + y_3 - h)

        vertices = np.array([c1, c2, c3], np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(image, [vertices], color=colour)

        return image


    def draw_circle(self, image, quadrant, colour,size=""):

        points = {Size.LARGE: 45, Size.SMALL: 20}

        r = points[size]
      
        offset_x , offset_y = self.get_offset(quadrant)

        cv2.circle(image,(offset_x + 50, offset_y + 50), r, colour, -1) 

        return image 


    def generate_random_image(self):
        image_map = [0,0,0,0,0,0,0,0,0]
        
        shapes = [Shape.CIRCLE,Shape.TRIANGLE,Shape.SQUARE,""]
        colours = [self.blue,self.green,self.red]
        sizes = [Size.SMALL,Size.LARGE]

        for i in range(0,len(image_map)):
            image_map[i] = {"quad": i,
                            "shape": random.choice(shapes),
                            "colour": random.choice(colours),
                            "size":random.choice(sizes)}


        return image_map


    def generate_image(self,rule,negated,label,id):
        self.label = []
        image =  np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rule_satisified = False
        image_map = np.zeros(9)

        ## Generates images by checking if image satisfies the rule 
        while not rule_satisified:
            image_map = self.generate_random_image()
            rule_satisified = not self.check_image(image_map,rule,negated)

        for i in range(0,len(image_map)):
            image = self.draw_shape(image_map[i],image)

        
        ## Writes labels and writes to file
        labels_text = '\n'.join(','.join(l) for l in self.label)

        folderpath = os.getcwd() + "/" +  self.filepath + "/" + label + "_s"+ str(id)
        os.makedirs(folderpath, exist_ok=True)


        filename = "labels.txt"
        file_path = os.path.join(folderpath, filename)

        with open(file_path, 'w') as file:
            file.write(labels_text)


        filename = label + "_" + str(id) + ".png"
        file_path = os.path.join(folderpath, filename)
        cv2.imwrite(file_path, image)

    def change_filepath(self,filepath):
        self.filepath = filepath



    ####################################################GETTERS####################################################     

    def get_offset(self,qudarant):
        q = {0: (0,0)  , 1: (100,0)  , 2: (200,0),
             3: (0,100), 4: (100,100), 5: (200,100), 
             6: (0,200), 7: (100,200), 8: (200,200)
             }

        return q[qudarant]
        

    def get_shape(self,rule): 

        if "triangle" in rule:
            return Shape.TRIANGLE
        elif "square" in rule:
            return Shape.SQUARE
        elif "circle" in rule:
            return Shape.CIRCLE
        else:
             return None
        

    def get_label(self,shape,colour,q,size):

        if colour[0] == 255:
            colour = "Blue"
        elif colour[1] == 255:
            colour = "Green"
        else:
            colour = "Red"

        if shape == "":
            self.label.append([str(q),"","",""])
            return

        label = [str(q)]
        label.append(shape)
        label.append(colour)
        label.append(size)

        self.label.append(label)
        
    def get_colour(self,rule): 

        if "red"  in rule:
            return self.red
        elif "green" in rule:
            return self.green
        elif "blue" in rule:
            return self.blue
        else:
            return None
        
    def get_position(self,rule): 
        if "left" in rule:
            return Position.LEFT
        elif "right" in rule:
            return Position.RIGHT
        elif "above" in rule:
            return Position.ABOVE
        elif "below" in rule:
                return Position.BELOW
        else: 
            return None
        
    def get_size(self,rule): 
        if "small" in rule:
            return Size.SMALL
        elif "large" in rule:
            return Size.LARGE
        else: 
            return None

class Size(Enum):
    LARGE = 1
    SMALL = 2
class Shape(Enum):
    SQUARE = 1
    TRIANGLE = 2
    CIRCLE = 3
class Rule(Enum):
    BASIC = 1 
    Postional = 2
    COMPOUND = 3
class Position(Enum):
    LEFT = 1
    RIGHT = 2 
    ABOVE = 3
    BELOW = 4
class Colour(Enum):
    Red = 1
    Green = 2
    Blue = 3
class SHAPESDATASET(Dataset):
    def __init__(
        self,
        data_dir: str = "datasets/SHAPES/training_data",
        transform=None,
        target_transform=None,
        cache: bool = True,
    ):
        super().__init__()
        self.cache = cache
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.num_slots = 10
        self.classes = {"shape": 4,"colour": 4, "size": 3}
        self.label_to_index = self.get_index(self._get_all_labels())
        


        self.dataset = DatasetFolder(
            root=data_dir,
            loader=lambda x: Image.open(x).convert("RGB"),
            extensions=".png",  
            transform=ToTensor(),  
        )

        if self.cache:
            from concurrent.futures import ThreadPoolExecutor

            self._images = []
            with ThreadPoolExecutor() as executor:
                self._images = list(
                    tqdm(
                        executor.map(self._load_image, self.dataset.samples),
                        total=len(self.dataset.samples),
                        desc=f"Caching Custom Dataset",
                        mininterval=(0.1 if os.environ.get("IS_NOHUP") is None else 90),
                    )
                )

    def get_index(self,labels):

        lab_to_idx = []
        offset = 0

        for l in labels:
            lab_to_idx.append({label: index + offset for index, label in enumerate(list(l))})
            offset += len(l)

        return lab_to_idx


    def _load_image(self, sample):
        image_path, label = sample[0], sample[1]

        labels_file_path = os.path.join(os.path.dirname(image_path), 'labels.txt')
        
        
        with open(labels_file_path, 'r') as file:
            labels = file.read().splitlines()


        labels = [tuple(map(str, t.split(',')[1:])) for t in labels]
        labels.append(("","",""))


        img_label = []
        
        for _ in range(0,self.num_slots):
            slot_labels = np.zeros(sum(self.classes.values()))

            img_label.append(slot_labels)

        final_labels = []
        
        for i in range(0,self.num_slots):
            i_label = labels[i]
            m_label = img_label[i]


            idx = [self.label_to_index[0][i_label[0]],self.label_to_index[1][i_label[1]],self.label_to_index[2][i_label[2]]]
  
            for j in idx:
                m_label[j] = 1

            final_labels.append(np.array(m_label, dtype=np.int64))
            
        img_label = np.stack(final_labels)
        return image_path, np.array(img_label,dtype=np.int64)
    
    def __len__(self):
        return len(self.dataset)
    
    def _get_all_labels(self):
        shape_labels = ["","Triangle","Circle","Square"]
        colour_labels = ["","Red","Green","Blue"]
        size_labels = ["","L","S"]

        all_labels = [shape_labels, colour_labels, size_labels] 
    
        return all_labels

    def __getitem__(self, idx: int):
        if self.cache:
            image_path, label = self._images[idx]
        else:
            image_path, label = self.dataset.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label