from sklearn.metrics import adjusted_rand_score
from datasets.SHAPES_4.SHAPES4 import SHAPESDATASET
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from neuro_modules.NAL import NAL
import pipelines.shapes as shapes


@torch.no_grad()
def visualise_slots(model, img_path, num_slots, idx=0):
    img_path = NAL.preprocess_image(None,img_path)
    recon_combined, recons, masks, slots, _  = model(img_path)
    image = renormalize(img_path)[idx]
    recon_combined = renormalize(recon_combined)[idx]
    recons = renormalize(recons)[idx]
    masks = masks[idx]
    
    image = to_numpy(image.permute(1,2,0))
    recon_combined = to_numpy(recon_combined.permute(1,2,0))
    recons = to_numpy(recons.permute(0,2,3,1))
    masks = to_numpy(masks.permute(0,2,3,1))
    slots = to_numpy(slots)

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

    plt.savefig('slot_vis_shape.png')


def analyse_SHAPES():
    DATA_DIR_TRAIN = "datasets/SHAPES_9/training_data/"
    NUM_CLASSESS = 10
    SAMPLES = 200


    data_train = shapes.get_SHAPES_dataset(DATA_DIR_TRAIN,SAMPLES,NUM_CLASSESS)

    stats = {"Triangle" : {"Red" : 0, "Green": 0, "Blue": 0},
             "Square" : {"Red" : 0, "Green": 0, "Blue": 0},
             "Circle" : {"Red" : 0, "Green": 0, "Blue": 0},
             "" : {"" : 0}}
    
    for i in range (1,11):
        for _,img_path in data_train[i]:
            for label in shapes.get_ground_truth(img_path):
                stats[label[1]][label[2]] += 1


    print("---------Shape-----------")
    print(f"Total Triangles : {sum(stats['Triangle'].values())} ")
    print(f"Total Squares : {sum(stats['Square'].values())} ")
    print(f"Total Circles : {sum(stats['Circle'].values())} ")
    print(f"Total Blanks: {sum(stats[''].values())} ")

    print("---------Colour-----------")
    green_shapes = stats["Circle"]["Green"] + stats["Triangle"]["Green"] + stats["Square"]["Green"]
    red_shapes = stats["Circle"]["Red"] + stats["Triangle"]["Red"] + stats["Square"]["Red"]
    blue_shapes = stats["Circle"]["Blue"] + stats["Triangle"]["Blue"] + stats["Square"]["Blue"]
    print(f"Total Green Shapes: {green_shapes}")
    print(f"Total Red Shapes: {red_shapes}")
    print(f"Total Blue Shapes: {blue_shapes}")


def analyse_CLEVR():
    pass


def assign_pixels_to_clusters(image, attention_masks, background_value=0):
    assigned_clusters = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Skip background pixels
            if all(image[y, x] == background_value):
                assigned_clusters.append(-1)
                continue
            
            pixel_value = image[y, x]
            max_attention = 0
            assigned_cluster = None
            for cluster_id, attention_mask in enumerate(attention_masks):
                attention_value = attention_mask[y, x]
                if attention_value > max_attention:
                    max_attention = attention_value
                    assigned_cluster = cluster_id
            assigned_clusters.append(assigned_cluster)
    return assigned_clusters



def get_ari_score(true_map, predicted_map):
    
    true_labels = true_map.flatten()
    predicted_labels = predicted_map.flatten()

    ari_score = adjusted_rand_score(true_labels, predicted_labels)

    return ari_score

####### Helper Functions ##########


def to_numpy(x):
    return x.cpu().detach().numpy()

def renormalize(x):
    return x / 2. + 0.5  # [-1, 1] to [0, 1]


if __name__ == "__main__":  
    analyse_SHAPES()