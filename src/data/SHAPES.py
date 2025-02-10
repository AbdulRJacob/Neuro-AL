import copy
import os
import re
import random
from enum import Enum

import numpy as np
import cv2
import yaml
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor
from sklearn.preprocessing import OneHotEncoder

def load_config(config_path="config/shapes_config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

SHAPE_CONFIG = load_config()

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
                for j in range(len(obj_id)): ## obj_id corresponds to each positional rule element  ## TODO: Bug with index taking objs from pre exceptions
                    entry[obj_id[j]]["shape"] = self.get_shape(rules[j])
                    entry[obj_id[j]]["colour"] = self.get_colour(rules[j])
                    entry[obj_id[j]]["size"] = self.get_size(rules[j])

                output.append((parsed_rule[i][0], entry, obj_id, parsed_rule[i][2],parsed_rule[i][-1]))

        return output
    

    def check_image(self,image,rule,negated,idx=0):
        
        if idx >= len(rule):
            return True

        r, r_obj, obj_id , pos, negated_rule = rule[idx]

        negated = negated ^ negated_rule
      
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
        loc = (0,0)
        quad, shape, colour, size = list(grid_info.values())

        if colour == "":
            colour = self.get_colour(colour)

        if shape == Shape.TRIANGLE:
            image ,centrioid = self.draw_triangle(image, quad, colour,size)
            s = "Triangle"
            loc = centrioid

        elif shape == Shape.SQUARE:
            image ,centrioid = self.draw_square(image, quad, colour,size)
            s = "Square"
            loc = centrioid


        elif shape == Shape.CIRCLE:
            image ,centrioid = self.draw_circle(image, quad, colour,size)
            s = "Circle"
            loc = centrioid


        if size == Size.LARGE:
            sz = "L"
        else:
            sz = "S"
        
        self.get_label(s,colour,loc,sz)

        return image
    
    def draw_square(self, image, quadrant, colour, size="", attention_map = None, get_map=False):
        points = {Size.LARGE: (5,5,95,95), Size.SMALL: (30,30,70,70)}

        (x_1,y_1,x_2,y_2) = points[size]

        offset_x , offset_y = self.get_offset(quadrant)

        center_x = (offset_x + x_1 + offset_x + x_2) // 2
        center_y = (offset_y + y_1 + offset_y + y_2) // 2

        centroid = (center_x,center_y)


        if get_map:
             attention_map[offset_y + y_1:offset_y + y_2, offset_x + x_1:offset_x + x_2] = quadrant + 1

             return attention_map

        cv2.rectangle(image, (offset_x + x_1, offset_y + y_1), (offset_x + x_2, offset_y + y_2), colour, -1) 

        return image , centroid
    
    def draw_triangle(self, image, quadrant, colour,size="", attention_map=None, get_map=False):

        points = {Size.LARGE: (5,95,95,95,50,95,90), Size.SMALL: (30,70,70,70,50,70,40)}

        (x_1,y_1,x_2,y_2,x_3,y_3,s) = points[size]
    
        offset_x , offset_y = self.get_offset(quadrant)
       
        c1 = (offset_x + x_1, offset_y + y_1)
        c2 = (offset_x + x_2, offset_y + y_2)

        h = int((3**0.5 / 2) * s) 
        c3 = (offset_x + x_3, offset_y + y_3 - h)

        vertices = np.array([c1, c2, c3], np.int32).reshape((-1, 1, 2))

        centroid_x = (c1[0] + c2[0] + c3[0]) / 3
        centroid_y = (c1[1] + c2[1] + c3[1]) / 3
        centroid = (centroid_x, centroid_y)

        if get_map:
            cv2.fillPoly(attention_map, [vertices], color=quadrant + 1)
            return attention_map

        cv2.fillPoly(image, [vertices], color=colour)

        return image, centroid


    def draw_circle(self, image, quadrant, colour, size="", attention_map=None, get_map=False ):

        points = {Size.LARGE: 45, Size.SMALL: 20}

        r = points[size]
      
        offset_x , offset_y = self.get_offset(quadrant)

        center_x = offset_x + 50
        center_y =  offset_y + 50

        if get_map:
            cv2.circle(attention_map, (center_x, center_y), r, quadrant + 1, -1) 
            return attention_map

        cv2.circle(image,(offset_x + 50, offset_y + 50), r, colour, -1) 

        return image, (center_x,center_y)


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
    
    def generate_attention_map(self,labels): 
        map = np.zeros((300,300), dtype=np.uint8)
        image =  np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for label in labels:
            x, y, shape,  _ , size = label
            quad = self.find_section(x,y)
            size = "small" if size == "S" else "large"
            if shape == "Square":
                map = self.draw_square(image,int(quad),"",self.get_size(size),map,True)
            elif shape == "Circle":
                map = self.draw_circle(image,int(quad),"",self.get_size(size),map,True)
            elif shape == "Triangle":
                map = self.draw_triangle(image,int(quad),"",self.get_size(size),map,True)

        return map


    def generate_image(self,rule,negated,label,id):
        self.label = []
        image =  np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rule_satisified = False
        image_map = np.zeros(9)

        ## Generates images by checking if image satisfies the rule 
        while not rule_satisified:
            image_map = self.generate_random_image()
            rule_satisified = self.check_image(image_map,rule,negated)

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
    
    def find_section(self,x, y):
        row = y // 100
        col = x // 100
        section = row * 3 + col
        return section
        

    def get_shape(self,rule): 

        if "triangle" in rule:
            return Shape.TRIANGLE
        elif "square" in rule:
            return Shape.SQUARE
        elif "circle" in rule:
            return Shape.CIRCLE
        else:
             return None
        

    def get_label(self,shape,colour,loc,size):

        if colour[0] == 255:
            colour = "Blue"
        elif colour[1] == 255:
            colour = "Green"
        else:
            colour = "Red"

        if shape == "":
            return

        label = []
        label.append(str(loc[0]))
        label.append(str(loc[1]))
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
        data_dir: str = SHAPE_CONFIG["data"]["data_dir"],
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
        self.num_object = 9
        self.classes = {"shape": 3,"colour": 3, "size": 2}        


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


        labels_arr = [list(map(str, t.split(','))) for t in labels]

        if len(labels_arr) == 0:
            return  image_path , np.zeros((self.num_slots,11))

        all_labels = self._get_all_labels()
        feature_list = [l[2:] for  l in labels_arr]

        loc_list = np.array([l[:2] for  l in labels_arr], dtype=float) / 300

        encoder = OneHotEncoder(categories=all_labels,sparse_output=False)
        one_hot_encoded = encoder.fit_transform(feature_list)

        result = np.hstack((loc_list, one_hot_encoded))

        is_real = np.ones((result.shape[0], 1)) 
        result_with_real = np.hstack((result, is_real))

        # Paading
        pad_rows = self.num_slots - result_with_real.shape[0]
        pad_cols = 0  

        final_labels = np.pad(result_with_real, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

        class_label = self.extract_class_number(image_path)

        return {"input": image_path, "class": class_label, "target": final_labels}
    
    def __len__(self):
        return len(self.dataset)
    
    def _get_all_labels(self):
        shape_labels = ["Triangle","Circle","Square"]
        colour_labels = ["Red","Green","Blue"]
        size_labels = ["L","S"]

        all_labels = [shape_labels, colour_labels, size_labels] 
    
        return all_labels

    def extract_class_number(self,path):
        class_label = np.zeros(13)
        parts = path.split('/')

        class_directory = parts[-2]

        class_number = int(class_directory.split('_')[0][1:])

        class_label[class_number] = 1

        return class_label

    def __getitem__(self, idx: int):
        if self.cache:
            info = self._images[idx]
            image_path = info["input"]
            class_label = info["class"]
            label = info["target"]
        else:
            info = self._load_image(self.dataset.samples[idx])
            image_path = info["input"]
            class_label = info["class"]
            label = info["target"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return {"input": image, "class": class_label, "target": label}
    
    def get_SHAPES_dataset(self,split="train"):
        num_classes = 12
        dataset = {}

        if split == "train":
            start = 0
            stop = 1000
            data_dir = self.data_dir + "training_data/"

        if split == "test":
            start = 1000
            stop = 1500
            data_dir = self.data_dir + "testing_data/"
        
        for i in range(1, num_classes + 1) :
            dataset[i] = []
            for j in range(start,stop):
                img_path = data_dir + f"c{i}_s{j}/c{i}_{j}.png"
                label = data_dir + f"c{i}_s{j}/labels.txt"
                dataset[i].append((img_path,label))
        
        return dataset

    def get_ground_truth(self, label_path:str ):
        with open(label_path, "r") as file:
            lines = file.readlines()

            set_result = []
            for line in lines:
                item = line.strip().split(",") 
                if len(item) >= 3: 
                    info = [float(item[0]),float(item[1])]
                    info = info + item[2:]
                    set_result.append((tuple(info)))
                
            return set_result
        
    def get_ground_truth_map(self, label_path: str,):
        labels = self.get_ground_truth(label_path)

        generator = SHAPES(300,300,"")

        map = generator.generate_attention_map(labels)

        map = cv2.resize(map, (64,64),interpolation=cv2.INTER_NEAREST)

        return map
    
    def get_all_data(self):
        all_data_list = []
        for idx in range(self.__len__()):
            data = self.__getitem__(idx)
            all_data_list.append((idx, data))
        return all_data_list
    
    def get_transform():
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])
    
        return transform
    


def generate_dataset_SHAPES(rule, label, num_of_images, isPostive):
    base_dir = SHAPE_CONFIG["data"]["data_dir"]
    dir = [base_dir + "training_data/", base_dir + "testing_data/"]

    for directory in dir:
        if not os.path.exists(directory):
            os.makedirs(directory)

    
    num_train, num_test = num_of_images
    ## Generate Training Data
    shape_generator = SHAPES(300,300,dir[0]) 

    shape_generator.generate_shape(rule, isPostive, label, num_train)

    shape_generator.change_filepath(dir[1])

    shape_generator.generate_shape(rule, isPostive, label, num_test, num_train)


    
if __name__ == "__main__":
    
   train_test_split = (10,10)
   labels = [["c1","c2"],["c3","c4"],["c5","c6"],["c7","c8"],["c9","c10"],["c11","c12"]]
   rules = [["c(A) :- blue(O1), square(O1), in(A,O1), image(A)"],
            ["c(A) :- green(O1), triangle(O1), small(O1) in(A,O1), image(A)"],
            ["c(A) :- blue(O1), triangle(O1), in(A,O1), red(O2), circle(O2), large(O2) in(A,O2), image(A)"],
            ["c(A) :- blue(O1), square(O1), in(A,O1), green(O2), triangle(O2), in(A,O2), above(O1,O2), image(A)"],
            ["c(A) :- red(O1), triangle(O1), in(A,O1), green(O2), circle(O2), small(O2), in(A,O2), left(O1,O2), image(A)"],
            ["c(A) :- not exception1(A), image(A).", "exception1(A) :- blue(O1), circle(O1),in(A,O1), image(A)."]]
   
   for i in range(0,len(labels)):
        rule = rules[i] 
        label = labels[i]
        l1 = label[0]
        l2 = label[1]

        pos1 = int(l1.replace("c","")) % 2 == 0
        pos2 = int(l2.replace("c","")) % 2 == 0

        generate_dataset_SHAPES(rule,l1,train_test_split,pos1)
        generate_dataset_SHAPES(rule,l2,train_test_split,pos2)
