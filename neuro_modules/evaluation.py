from sklearn.metrics import adjusted_rand_score
from datasets.SHAPES_9.SHAPES import SHAPESDATASET
from datasets.SHAPES_4.SHAPES4 import SHAPESDATASET as SHAPESDATASET_4
import torch
import os
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from neuro_modules.NAL import NAL
import neuro_modules.utils as utils
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

def visualise_clustering_maps(clustering_map,true_map):
    

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(clustering_map, cmap='viridis')
    axes[0].imshow(clustering_map, cmap='viridis')
    axes[0].set_title('Clustering Map')
    axes[0].set_xlabel('Width')  
    axes[0].set_ylabel('Height')  
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(true_map, cmap='viridis')
    axes[1].imshow(true_map, cmap='viridis')
    axes[1].set_title('A Map')
    axes[1].set_xlabel('Width')  
    axes[1].set_ylabel('Height')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig('clustering_map.png')


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


def calcuate_ari_resutls(visualise=False):
    
    num_slots = 5
    dataset = SHAPESDATASET(cache=False)
    all_data = [item for sublist in dataset.get_SHAPES_dataset().values() for item in sublist]

    total_ari = 0.0
    count = 0.0

    for (img_path,label_path) in all_data:
        if not os.path.isfile(img_path):
            continue

        nal = shapes.get_shape_9_nal_model()

        pred, mask, = nal.run_slot_attention_model(img_path,num_slots)
        mask = mask.squeeze(1).numpy()

        image = Image.open(img_path).resize((64, 64))
        image_array = np.array(image)

        ## Calcualating ARI

        pixel_assignments = utils.assign_pixels_to_clusters(image_array,mask)
        true_pixel_assignments = dataset.get_ground_truth_map(label_path).flatten()
        

        ari_score = adjusted_rand_score(true_pixel_assignments,pixel_assignments)
        total_ari+= ari_score
        count+=1

    if visualise:
        clustering_map = np.array(pixel_assignments).reshape((64,64))
        true_map = true_pixel_assignments.reshape((64, 64))
        visualise_clustering_maps(clustering_map,true_map)


    return total_ari / count


def get_ari_score(true_map, predicted_map):
    
    true_labels = true_map.flatten()
    predicted_labels = predicted_map.flatten()

    ari_score = adjusted_rand_score(true_labels, predicted_labels)

    return ari_score


def calcualte_AP():
    transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])
    dataset = SHAPESDATASET(transform=transform)
    loader = DataLoader(dataset,len(dataset),num_workers=2)

    nal = shapes.get_shape_9_nal_model()
    model= nal.model


    for t, (x, y) in enumerate(loader):
        x = (x / 127.5 ) - 1
        y = y.float()
        _ , _ ,_ ,_, y_hat = model(x)
        ap = utils.average_precision(y_hat,y,-1)

    return ap

####### Helper Functions ##########


def to_numpy(x):
    return x.cpu().detach().numpy()

def renormalize(x):
    return x / 2. + 0.5  # [-1, 1] to [0, 1]



if __name__ == "__main__":  
    print(calcuate_ari_resutls())

