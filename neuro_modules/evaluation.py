from sklearn.cluster import KMeans
from datasets.SHAPES_4.SHAPES4 import SHAPESDATASET
from models.slots_shapes4 import SlotAutoencoder
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def to_numpy(x):
    return x.cpu().detach().numpy()

def renormalize(x):
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

    num_slots = self.num_of_slots
    fig, ax = plt.subplots(1, num_slots + 2, figsize=(18, 4))
    ax[0].imshow(image)
    ax[0].set_title('Input')
    ax[1].imshow(recon_combined)
    ax[1].set_title('Reconstruction')
    for i in range(num_slots):
        ax[i + 2].imshow(recons[i] * masks[i] + (1 - masks[i]))
        ax[i + 2].set_title('Slot %s' % str(i + 1))
    for i in range(len(ax)):
        ax[i].grid(False)
        ax[i].axis('off')

    plt.savefig('slot_vis.png')



if __name__ == "__main__":  
    pass