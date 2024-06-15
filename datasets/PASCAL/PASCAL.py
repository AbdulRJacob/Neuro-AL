import os
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
from tqdm import tqdm
import numpy as np

class PascalVOC(Dataset):
    def __init__(self, data_dir="", year='2012', split='train', transform=None, cache=False):
        self.data_dir = data_dir
        self.year = year
        self.split = split
        self.transform = transform
        self.cache = cache
        self.labels = []  # To store labels
        self.classes = self._load_classes()
        self.label_to_index = {label: index for index, label in enumerate(self.classes)}
        self.one_hot_encoder = OneHotEncoder(categories=[list(range(len(self.classes)))], sparse_output=False)
        
        # Load dataset
        self.image_paths, self.labels = self._load_dataset()

        if self.cache:
            self._data = []
            with ThreadPoolExecutor() as executor:
                self._data = list(
                    tqdm(
                        executor.map(self._load_data, range(len(self.labels))),
                        total=len(self.labels),
                        desc=f"Caching Pascal VOC Dataset",
                        mininterval=0.1,
                    )
                )

    def _load_classes(self):
        return ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
                "train", "tvmonitor"]

    def _load_dataset(self):
        image_paths = []
        labels = []

        # Read image sets
        split_file = os.path.join(self.data_dir, f'VOC{self.year}/ImageSets/Main/{self.split}.txt')
        with open(split_file, 'r') as f:
            image_ids = f.read().strip().split()

        for image_id in image_ids:
            img_path = os.path.join(self.data_dir, f'VOC{self.year}/JPEGImages/{image_id}.jpg')
            ann_path = os.path.join(self.data_dir, f'VOC{self.year}/Annotations/{image_id}.xml')
            image_paths.append(img_path)
            labels.append(self._process_annotation(ann_path))

        return image_paths, labels

    def _process_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        boxes = []
        classes = []

        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in self.classes:
                continue
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(self.label_to_index[cls])

        classes_one_hot = self.one_hot_encoder.fit_transform(np.array(classes).reshape(-1, 1)).astype(int)
        return {'boxes': np.array(boxes), 'classes': classes_one_hot}

    def _load_data(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        return {'input': img, 'class': label['classes'], 'boxes': label['boxes']}

    def __getitem__(self, idx):
        if self.cache:
            return self._data[idx]
        else:
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            label = self.labels[idx]
            return {'input': img, 'class': label['classes'], 'boxes': label['boxes']}

    def __len__(self):
        return len(self.labels)
    
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = PascalVOC(data_dir='/mnt/d/fyp/Pascal_VOC/VOCdevkit/', year='2012', split='val', transform=transform, cache=False)
    
    # Fetch a sample
    info= dataset[0]
    print(info["input"].shape)