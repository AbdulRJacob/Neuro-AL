import copy
import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import json
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class CLEVR(Dataset):
    def __init__(self, data_dir, split='train', transform=None, cache=True):
        self.data_dir =  data_dir # "/mnt/d/fyp/CLEVR_v1.0/CLEVR_v1.0"
        self.split = split
        self.transform = transform
        self.cache = cache
        self.labels = []  # To store labels
        self.num_slots = 11
        self.classes = {"shape": 4,"color": 9, "size": 3, "material": 3}
        self.label_to_index = self.get_index(self._get_all_labels())
        
        # Load labels
        data_file = os.path.join(self.data_dir, f'scenes/CLEVR_{self.split}_scenes.json')
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.labels = [scene['objects'] for scene in data['scenes']]
        
        # Load images and cache if required
        self.image_paths = [os.path.join(self.data_dir, f'images/{self.split}/CLEVR_{self.split}_{str(i).zfill(6)}.png') for i in range(len(self.labels))]

        if self.cache:
            self._data = []
            with ThreadPoolExecutor() as executor:
                self._data = list(
                    tqdm(
                        executor.map(self._load_data, range(len(self.labels))),
                        total=len(self.labels),
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

    def _get_all_labels(self):
        shape_labels = ["", "cube", "sphere", "cylinder"]
        colour_labels = ["", "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
        size_labels = ["", "large","small"]
        material_labels = ["","rubber","metal"]

        all_labels = [shape_labels, colour_labels, size_labels, material_labels]

        return all_labels


    def process_scene(self,scene_info):
        
        img_label = []

        for _ in range(0,self.num_slots):
            slot_labels = np.zeros(sum(self.classes.values()))

            img_label.append(slot_labels)

        final_labels = []
        
        for i in range(0,self.num_slots):
            m_label = img_label[i]

            if i < len(scene_info):
                i_label = scene_info[i]

                idx = []

                for j in enumerate(list(self.classes.keys())):
                    idx.append(self.label_to_index[j[0]][i_label[j[1]]])
    
                for k in idx:
                    m_label[k] = 1

            final_labels.append(np.array(m_label, dtype=np.int64))
            
        img_label = np.stack(final_labels)

        return img_label

    def _load_data(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.process_scene(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __getitem__(self, idx):
        if self.cache:
            img, label = self._data[idx]
        else:
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
            if self.transform is not None:
                img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)

