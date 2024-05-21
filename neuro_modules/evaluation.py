from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datasets.SHAPES_9.SHAPES import SHAPESDATASET
from datasets.SHAPES_4.SHAPES4 import SHAPESDATASET as SHAPESDATASET_4
import torch
import os
from PIL import Image
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
    
    num_slots = 10
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

        fakes = []

        for i,item in enumerate(pred):
            if item[1]['real'] < 0.5:
                fakes.append(i)

        fakes.sort(reverse=True)

        # Remove the specified indices
        for i in fakes:
            mask = np.delete(mask, i, axis=0)

        image = Image.open(img_path).resize((64, 64))
        image_array = np.array(image)

        ## Calcualating ARI

        pixel_assignments = utils.assign_pixels_to_clusters(image_array,mask).flatten()
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


def nal_eval_metrics():
    classes= [1,2]
    dataset = SHAPESDATASET(cache=False).get_SHAPES_dataset()
    
    aba_path = "shapes_9_bk_SOLVED.aba"

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for (img_path, _) in dataset[classes[0]]:

        if not os.path.isfile(img_path):
            continue
    
        prediction = shapes.shapes_9_nal_inference(img_path,aba_path)

        if prediction:
            tp+=1
        else:
            fn+=1
        

    for (img_path, _) in dataset[classes[1]]:

        if not os.path.isfile(img_path):
            continue
    
        prediction = shapes.shapes_9_nal_inference(img_path,aba_path)

        if prediction:
            fp+=1
        else:
            tn+=1

    print(tp,tn,fp,fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

    matrix = confusion_matrix([1, 0], [1, 0])  # Creating an empty matrix of size 2x2
    matrix[0][0] = tp
    matrix[1][1] = tn
    matrix[0][1] = fp
    matrix[1][0] = fn

    # Create a heatmap for visualization
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Positive', 'Predicted Negative'], yticklabels=['Actual Positive', 'Actual Negative'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

####### Helper Functions ##########


def to_numpy(x):
    return x.cpu().detach().numpy()

def renormalize(x):
    return x / 2. + 0.5  # [-1, 1] to [0, 1]



if __name__ == "__main__":  
    print(calcuate_ari_resutls(True))
