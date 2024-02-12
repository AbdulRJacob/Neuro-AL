import copy
import os
import random
from typing import Callable, Optional, Tuple
from enum import Enum

import numpy as np
import cv2
from pyparsing import one_of, OneOrMore, Optional, ParseException, Combine, CaselessLiteral, delimitedList
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


    def generate_shape(self, rule: str, label: str, batch_size: int):
      
        print(f"Generating Data for rule: \"{rule.lower()}\"")

        parsed_rules = self.parse_rule(rule.lower())
        
        for i in range(batch_size):
            if (i % 500 == 0):
                print(f"Generated {i}/{batch_size} images")

            if (parsed_rules == []):
                print("Invalid Rule")
                return 0
            
            self.generate_image(parsed_rules,label,i)
        
        print("Done!")
        return 1
    

    def parse_rule(self, rule: str): 
        ## Grammar
        shape = one_of(["square","circle","triangle"], caseless=True)
        colour = one_of(["red","green","blue"] ,caseless=True)
        pos = one_of(["left","right","above","below"], caseless=True)
        con = CaselessLiteral("and") | CaselessLiteral("a") | CaselessLiteral("contains") | CaselessLiteral("of")


        nonSpacialGrammar = Combine(con + " " + Optional(colour + " ") + shape)
        nonSpatialRule = delimitedList(nonSpacialGrammar, delim=con)


        spatialGrammar = con + Optional(colour) + shape + pos + Optional(con) + Optional(colour) + shape
        spatialRule = delimitedList(spatialGrammar, delim=con)

        
        compoundSpatialGrammar = con + Optional(colour) + shape + pos + Optional(con) + shape + OneOrMore(con + pos + Optional(colour) + Optional(con) + shape)
        compoundspatialRule = delimitedList(compoundSpatialGrammar, delim=con)


        parsed_rule = []

        try:
            parsed_rule.append((Rule.COMPOUND,compoundspatialRule.parseString(rule, parse_all=True) ))
        except:
            split_rules = rule.split("and")

            for split_rule in split_rules:
                try:
                    parsed_rule.append((Rule.SPATIAL,spatialRule.parseString(split_rule, parse_all=True)))
                except:
                    try:
                        parsed_rule.append((Rule.BASIC,nonSpatialRule.parseString(split_rule, parse_all=True)))
                    except:
                        parsed_rule = []


        output = []
        for i in range(len(parsed_rule)):
            obj = {"shape": "", "colour": "", "size": ""}
            if (parsed_rule[i][0] == Rule.BASIC):
                rule = parsed_rule[i][1][0]
                obj["shape"] = self.get_shape(rule)
                obj["colour"] = self.get_colour(rule)
                obj["size"] = self.get_size(rule)
                output.append((parsed_rule[i][0], obj))


        return output
    

    def check_image(self,image,rule):

        r, r_obj = rule[0]
        negated = True
        cond_satisfied = False
        
        if r == Rule.BASIC:
            for i in image:
                cond = i["shape"] == r_obj["shape"] and (i["colour"] == r_obj["colour"] or r_obj["colour"] == None ) and (i["size"] == r_obj["size"] or r_obj["size"] == None)
                if cond:
                    cond_satisfied = True

            if negated:
                return not cond_satisfied
            else:
                return cond_satisfied
    
    def draw_shape(self, grid_info,image):

        s = ""
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

        
        self.get_label(s,colour,quad,size)


        return image
    
    def draw_square(self, image, quadrant, colour, size=""):
        points = {"L": (5,5,95,95), "S": (30,30,70,70)}
        choice = ["L", "S"]

        (x_1,y_1,x_2,y_2) = points[random.choice(choice)]

        offset_x , offset_y = self.get_offset(quadrant)

        cv2.rectangle(image, (offset_x + x_1, offset_y + y_1), (offset_x + x_2, offset_y + y_2), colour, -1) 

        return image
    
    def draw_triangle(self, image, quadrant, colour,size=""):

        points = {"L": (5,95,95,95,50,95,90), "S": (30,70,70,70,50,70,40)}
        choice = ["L", "S"]

        (x_1,y_1,x_2,y_2,x_3,y_3,s) = points[random.choice(choice)]
    
        offset_x , offset_y = self.get_offset(quadrant)
       
        c1 = (offset_x + x_1, offset_y + y_1)
        c2 = (offset_x + x_2, offset_y + y_2)

        h = int((3**0.5 / 2) * s) 
        c3 = (offset_x + x_3, offset_y + y_3 - h)

        vertices = np.array([c1, c2, c3], np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(image, [vertices], color=colour)

        return image


    def draw_circle(self, image, quadrant, colour,size=""):

        points = {"L": 45, "S": 20}
        choice = ["L", "S"]

        r = points[random.choice(choice)]
      
        offset_x , offset_y = self.get_offset(quadrant)

        cv2.circle(image,(offset_x + 50, offset_y + 50), r, colour, -1) 

        return image 


    def generate_random_image(self):
        image_map = [0,0,0,0,0,0,0,0,0]
        
        shapes = [Shape.CIRCLE,Shape.TRIANGLE,Shape.SQUARE,""]
        colours = [self.blue,self.green,self.red]
        sizes = ["L","S"]

        for i in range(0,len(image_map)):
            image_map[i] = {"quad": i,
                            "shape": random.choice(shapes),
                            "colour": random.choice(colours),
                            "size":random.choice(sizes)}


        return image_map


    def generate_image(self,rule,label,id):
        self.label = []
        image =  np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rule_satisified = False
        image_map = np.zeros(9)

        
        while not rule_satisified:
            image_map = self.generate_random_image()
            rule_satisified = self.check_image(image_map,rule)

        for i in range(0,len(image_map)):
            image = self.draw_shape(image_map[i],image)

        
        labels_text = '\n'.join(','.join(l) for l in self.label)

        folderpath = os.getcwd() +  self.filepath + "/" + label + "_s"+ str(id)
        os.makedirs(folderpath, exist_ok=True)


        filename = "labels.txt"
        file_path = os.path.join(folderpath, filename)

        with open(file_path, 'w') as file:
            file.write(labels_text)


        filename = label + "_" + str(id) + ".png"
        file_path = os.path.join(folderpath, filename)
        cv2.imwrite(file_path, image)



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
            self.label.append([str(q),"",""])
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
    SPATIAL = 2
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
        data_dir: str = "datasets/training_data",
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
        self.classes = 9
        self.label_to_index = {label: index for index, label in enumerate(list(self._get_all_labels()))}
        


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



    def _load_image(self, sample):
        image_path, label = sample[0], sample[1]

        labels_file_path = os.path.join(os.path.dirname(image_path), 'labels.txt')
        
        
        with open(labels_file_path, 'r') as file:
            labels = file.read().splitlines()


        labels = [tuple(map(str, t.split(',')[1:])) for t in labels]
        labels.append(("","",""))


        img_label = [np.zeros(self.classes) for _ in range(0,self.num_slots)]

        for i in range(0,self.num_slots):
            label = labels[i]
            idx = [self.label_to_index[label[0]],self.label_to_index[label[1]]]
            for j in idx:
                img_label[i][j] = 1

        
        img_label = np.array(img_label, dtype=np.int64)

        return image_path, np.array(img_label, dtype=np.int64), np.ones((1,self.num_slots,self.classes))

    def __len__(self):
        return len(self.dataset)
    
    def _get_all_labels(self):
        all_labels = all_labels = ["","Triangle","Circle","Square","Red","Green","Blue","L","S"]
    
        return all_labels

    def __getitem__(self, idx: int):
        if self.cache:
            image_path, label, slots = self._images[idx]
        else:
            image_path, label, slots = self.dataset.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label