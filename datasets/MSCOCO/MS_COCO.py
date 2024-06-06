import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class COCODataset(Dataset):
    def __init__(self, data_dir, year="2017", transform=None, cache=False, split='train'):
        """
        Args:
            data_dir (string): Root directory where images are downloaded to.
            year (string): Year of the dataset (e.g., '2017').
            transform (callable, optional): Optional transform to be applied on a sample.
            cache (bool, optional): If True, cache the dataset in memory.
            split (string, optional): Dataset split, one of ['train', 'val', 'test'].
        """
        self.data_dir = os.path.join(data_dir, f'{split}{year}/{split}{year}')
        annFile = os.path.join(data_dir, f'annotations_trainval{year}/annotations/instances_{split}{year}.json')
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.cache = cache
        self.split = split

        if self.cache:
            self._data = []
            with ThreadPoolExecutor() as executor:
                self._data = list(
                    tqdm(
                        executor.map(self._load_data, range(len(self.ids))),
                        total=len(self.ids),
                        desc=f"Caching COCO {self.split} Dataset",
                    )
                )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.cache:
            return self._data[index]
        else:
            return self._load_data(index)

    def _load_data(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # Apply the transformation if it is defined
        if self.transform is not None:
            img = self.transform(img)

        # Prepare the annotations
        boxes = []
        labels = []
        for ann in anns:
            x, y, width, height = ann['bbox']
            boxes.append([x, y, x + width, y + height])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        return img, target
