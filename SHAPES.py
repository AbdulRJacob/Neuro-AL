import copy
import os
import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor

class SHAPES:

    def __init__(self) -> None:
        pass

    
    #TODO Recomplete Class







class SHAPESDATASET(Dataset):
    def __init__(
        self,
        data_dir: str = "training_data",
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
        self.classes = 7
        self.label_to_index = {label: index for index, label in enumerate(list(self._get_all_labels()))}
        


        self.dataset = DatasetFolder(
            root=data_dir,
            loader=lambda x: Image.open(x).convert("RGB"),
            extensions=".png",  # Adjust the extension based on your image format
            transform=ToTensor(),  # Adjust the transformation based on your needs
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
        labels.append(("",""))


        img_label = [np.zeros(self.classes) for _ in range(0,self.num_slots)]

        for i in range(0,self.num_slots):
            label = labels[i]
            idx = [self.label_to_index[label[0]],self.label_to_index[label[1]]]
            for j in idx:
                img_label[i][j] = 1

        
        img_label = np.array(img_label, dtype=np.int64)

        return image_path, np.array(img_label, dtype=np.int64)

    def __len__(self):
        return len(self.dataset)
    
    def _get_all_labels(self):
        all_labels = all_labels = ["Triangle","Circle","Square","Red","Green","Blue",""]
    
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