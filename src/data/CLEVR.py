import os
import yaml
import numpy as np
import tensorflow.compat.v1 as tf
import torch.nn as nn
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import OneHotEncoder


COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [240, 320]

MAX_NUM_ENTITIES = 11
BYTE_FEATURES = ['mask', 'image', 'color', 'material', 'shape', 'size']

features = {
    'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'z': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'pixel_coords': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'rotation': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'size': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'material': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'color': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}


def _decode(example_proto):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)


def load_config(config_path="config/clevr_config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

class CLEVR(Dataset):
    def __init__(self, data_dir = "", split='train', transform=None, cache=False):
        self.data_dir =  load_config()['data']['data_dir']
        self.split = split
        self.transform = transform
        self.cache = cache
        self.labels = []  # To store labels
        self.num_slots = 11
        self.classes = {"shape": 3, "color": 8, "size": 2, "material": 2}
        self.label_to_index = self.get_index(self._get_all_labels())
        
        # Load labels
        data_file = os.path.join(self.data_dir, f'scenes/CLEVR_{self.split}_scenes.json')
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.labels = [scene['objects'] for scene in data['scenes']][:1000]

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
        shape_labels = ["cube", "sphere", "cylinder"]
        colour_labels = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
        size_labels = ["large","small"]
        material_labels = ["rubber","metal"]

        all_labels = [shape_labels, colour_labels, size_labels, material_labels]

        return all_labels


    def process_scene(self,scene_info):
        
        labels_arr = [[d['shape'], d['color'], d['size'], d['material'], d["pixel_coords"]]for d in scene_info]
        all_labels = self._get_all_labels()
        feature_list = [l[:-1] for  l in labels_arr]

        loc_arr = np.array([l[-1] for  l in labels_arr], dtype=float)
        # Normalize x, y, z to [0, 1]
        normalized_x = loc_arr[:, 0] / 500.  # h
        normalized_y = loc_arr[:, 1] / 300.  # w 
        normalized_z = loc_arr[:, 2] / 16.   # d 

        # Combine the normalized coordinates into a single list (loc_list)
        loc_list = np.column_stack((normalized_x, normalized_y, normalized_z))

        # loc_list = (np.array([l[-1] for  l in labels_arr], dtype=float) + 3.) / 6

        encoder = OneHotEncoder(categories=all_labels,sparse_output=False)
        one_hot_encoded = encoder.fit_transform(feature_list)

        result = np.hstack((loc_list, one_hot_encoded))
        
        is_real = np.ones((result.shape[0], 1)) 
        result_with_real = np.hstack((result, is_real))

        # Paading
        pad_rows = self.num_slots - result_with_real.shape[0]
        pad_cols = 0  

        final_labels = np.pad(result_with_real, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

        return final_labels
    

    def _load_data(self, idx):
        img_path = self.image_paths[idx] if os.path.exists(self.image_paths[idx]) else self.image_paths[0]
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
            label = self.process_scene(self.labels[idx])
            if self.transform is not None:
                img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)
    


class CLEVRHans(Dataset):
    def __init__(self, data_dir = "", split='train', transform=None, cache=False):
        self.data_dir =  "datasets/CLEVR/"
        self.split = split
        self.transform = transform
        self.cache = cache
        self.labels = []  # To store labels
        self.num_slots = 11
        self.classes = {"shape": 3, "color": 8, "size": 2, "material": 2}
        self.num_id = 3
        self.label_to_index = self.get_index(self._get_all_labels())
        
        # Load labels
        data_file = os.path.join(self.data_dir, f'{self.split}/CLEVR_HANS_scenes_{self.split}.json')
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.labels = [(scene["class_id"], scene['objects']) for scene in data['scenes']]
        self.image_paths = [os.path.join(self.data_dir, f'{self.split}/images/{scene["image_filename"]}') for scene in data['scenes']]

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
        shape_labels = ["cube", "sphere", "cylinder"]
        colour_labels = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
        size_labels = ["large","small"]
        material_labels = ["rubber","metal"]

        all_labels = [shape_labels, colour_labels, size_labels, material_labels]

        return all_labels


    def process_scene(self,scene_info):
        

        labels_arr = [[d["3d_coords"], d['shape'], d['color'], d['size'], d['material']] for d in scene_info]
        all_labels = self._get_all_labels()
        feature_list = [l[1:] for  l in labels_arr]

        loc_list = (np.array([l[0] for  l in labels_arr], dtype=float) + 3.) / 6

        encoder = OneHotEncoder(categories=all_labels,sparse_output=False)
        one_hot_encoded = encoder.fit_transform(feature_list)

        result = np.hstack((loc_list, one_hot_encoded))

        is_real = np.ones((result.shape[0], 1)) 
        result_with_real = np.hstack((result, is_real))

        # Paading
        pad_rows = self.num_slots - result_with_real.shape[0]
        pad_cols = 0  

        final_labels = np.pad(result_with_real, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

        return final_labels
    

    def _load_data(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        cid, label = self.labels[idx]
        label = self.process_scene(label)
        if self.transform is not None:
            img = self.transform(img)

        return {"input": img, "class": np.eye(1, 3, int(cid), dtype=int).flatten(), "target": label}

    def __getitem__(self, idx):
        if self.cache:
            return self._data[idx]
        else:
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert("RGB")
            cid, label = self.labels[idx]
            label = self.process_scene(label)
            if self.transform is not None:
                img = self.transform(img)
        return {"input": img, "class": np.eye(1, 3, int(cid), dtype=int).flatten(), "target": label}
    
    def get_all_data(self):
        all_data_list = []
        for idx in range(len(self.labels)):
            data = self.__getitem__(idx)
            all_data_list.append((idx, data))
        return all_data_list

    def __len__(self):
        return len(self.labels)
    
    def get_transform():
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])
    
        return transform


