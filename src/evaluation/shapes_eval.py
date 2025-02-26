import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from sklearn.metrics import adjusted_rand_score

from data.SHAPES import SHAPESDATASET

import utils.utils as utils
import training.shapes_train_symbolic as shapes
import inference.shapes_inference as shapes_inf

transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])




def visualise_clustering_maps(clustering_map,true_map):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(clustering_map, cmap='viridis')
    axes[0].imshow(clustering_map, cmap='viridis')
    axes[0].set_xlabel('Width')  
    axes[0].set_ylabel('Height')  

    im2 = axes[1].imshow(true_map, cmap='viridis')
    axes[1].imshow(true_map, cmap='viridis')
    axes[1].set_xlabel('Width')  
    axes[1].set_ylabel('Height')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('clustering_map_clvr.png')


def analyse_SHAPES():
    dataset = SHAPESDATASET(cache=False)
    data_train = dataset.get_SHAPES_dataset()

    stats = {"Triangle" : {"Red" : 0, "Green": 0, "Blue": 0},
             "Square" : {"Red" : 0, "Green": 0, "Blue": 0},
             "Circle" : {"Red" : 0, "Green": 0, "Blue": 0},
             "" : {"" : 0}}
    
    size_stat = {"S": 0, "L":0}
    
    for i in range (1,11):
        for _,img_path in data_train[i]:
            for label in dataset.get_ground_truth(img_path):
                x, y, shape, color, size = label
                stats[shape][color] += 1
                size_stat[size] += 1


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

    print("---------Size-----------")
    print(f"Total Large Shapes: {size_stat['L']}")
    print(f"Total Small Shapes: {size_stat['S']}")
    


def calcuate_ari_resutls(visualise=False):
    
    num_slots = 10
    dataset = SHAPESDATASET(cache=False)
    all_data = [item for sublist in dataset.get_SHAPES_dataset().values() for item in sublist]
    total_ari = []

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
        total_ari.append(ari_score)

    if visualise:
        clustering_map = np.array(pixel_assignments).reshape((64,64))
        true_map = true_pixel_assignments.reshape((64, 64))
        visualise_clustering_maps(clustering_map,true_map)


    file = open("ari.txt", "w+")
 
    content = str(total_ari)
    file.write(content)
    file.close()


    return total_ari


def get_ari_score(true_map, predicted_map):
    
    true_labels = true_map.flatten()
    predicted_labels = predicted_map.flatten()

    ari_score = adjusted_rand_score(true_labels, predicted_labels)

    return ari_score

def display_ari(results):
    models = ["shapes_9"]
    means = [np.mean(r) for r in results]
    variances = [np.var(r) for r in results]

    plt.bar(models, means, yerr=variances, capsize=5)
    plt.ylabel('Mean ARI')
    plt.title('Mean ARI and Variance for Three Models')
    plt.savefig("ari.png")


def calcualte_AP():
    dataset = SHAPESDATASET(transform=transform)
    loader = DataLoader(dataset,128,num_workers=2)

    nal = shapes.get_shape_9_nal_model()
    model= nal.model

    for ldr in loader:
        print(ldr)
        x = ldr['input']
        y = ldr['target']

        x = (x / 127.5 ) - 1
        y = y.float()
        _ , _ ,_ ,_, y_hat = model(x)

        y_hat = y_hat.detach().numpy()

        ap = utils.average_precision(y_hat,y.numpy(),-1)
        break

    print("Average Precision: " + ap)


def shapes_nal_eval_metrics():
    classes= [1,2]
    dataset = SHAPESDATASET(cache=False).get_SHAPES_dataset(split="test")
    
    aba_path = "models/shapes/shapes_9_bk_r1_SOLVED.aba"

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for (img_path, _) in dataset[classes[0]]:

        if not os.path.isfile(img_path):
            continue
    
        prediction = shapes_inf.shapes_nal_inference(img_path,aba_path,include_pos=True)

        if prediction:
            tp+=1
        else:
            fn+=1
        

    for (img_path, _) in dataset[classes[1]]:

        if not os.path.isfile(img_path):
            continue
    
        prediction = shapes_inf.shapes_nal_inference(img_path,aba_path,include_pos=True)

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
    plt.savefig('confused_matrix_r4.png')



if __name__ == "__main__": 

    print("======= ANALYSING SHAPES DATASET ============")
    analyse_SHAPES()

    print("========== SHAPES ML METRICS ===============")
    shapes_nal_eval_metrics()

    print("========= AP AND ARI METRICS ============")
    # calcualte_AP()  # TODO Comment out CLEVR code to use for SHAPES in utils.average_precision
    calcuate_ari_resutls(True)
