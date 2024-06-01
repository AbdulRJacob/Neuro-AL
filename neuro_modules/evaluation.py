import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix

from datasets.SHAPES_9.SHAPES import SHAPESDATASET
from datasets.SHAPES_4.SHAPES4 import SHAPESDATASET as SHAPESDATASET_4
from datasets.CLEVR.CLEVR import CLEVRHans
from datasets.CLEVR.CLEVR import dataset as ds

from neuro_modules.NAL import NAL
import neuro_modules.utils as utils
import pipelines.shapes as shapes
import pipelines.clevr as clevr
import neuro_modules.slots as slots

transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])


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

    plt.savefig('slot_vis_clevr.png')

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



def ari_clevr():
    tf_records_path = ''
    dataset = ds(tf_records_path)

    clevr_nal = clevr.get_clevr_nal_model()
    num_slots = 11
    total_ari = []

    for element in dataset.take(1000): 
        image = Image.fromarray(element['image'].numpy())
        true_mask = element['mask'].numpy().squeeze(-1)

        _ , mask, = clevr_nal.run_slot_attention_model(image,num_slots, isPath=False)

        mask = mask.squeeze(1).numpy()
 
        resized_true_mask = np.zeros((11, 64, 64), dtype=np.uint8)

        for i in range(true_mask.shape[0]):
            pil_image = Image.fromarray(true_mask[i])
            
            pil_image_resized = pil_image.resize((64, 64), Image.BILINEAR)
            
            resized_true_mask[i] = np.array(pil_image_resized)


        ## Calcualating ARI

        resized_true_mask = np.expand_dims(resized_true_mask, axis=0)
        mask = np.expand_dims(mask, axis=0)

    
        batch_size, num_entries, height, width = resized_true_mask.shape
        masks_reshaped = (
            torch.reshape(torch.tensor(resized_true_mask), [batch_size, num_entries, height * width])
            .permute([0, 2, 1])
            .to(torch.float)
        )

        batch_size, num_entries, height, width = mask.shape
        pred_masks_reshaped = (
            torch.reshape(torch.tensor(mask), [batch_size, num_entries, height * width])
            .permute([0, 2, 1])
            .to(torch.float)
        )
        ari_no_background = utils.adjusted_rand_index(
            masks_reshaped[..., 1:], pred_masks_reshaped
        )


        total_ari.append(ari_no_background.item())

    
    file = open("ari_clevr.txt", "w+")
 
    # Saving the array in a text file
    content = str(total_ari)
    file.write(content)
    file.close()


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
    models = ["shapes_9","shapes_4"]
    means = [np.mean(r) for r in results]
    variances = [np.var(r) for r in results]

    plt.bar(models, means, yerr=variances, capsize=5)
    plt.ylabel('Mean ARI')
    plt.title('Mean ARI and Variance for Three Models')
    plt.savefig("ari.png")


def calcualte_AP():
    transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])
    dataset = SHAPESDATASET(transform=transform)
    loader = DataLoader(dataset,128,num_workers=2)

    nal = shapes.get_shape_9_nal_model()
    model= nal.model
    print("calculating")

    for t, (x, y) in enumerate(loader):
        x = (x / 127.5 ) - 1
        y = y.float()
        _ , _ ,_ ,_, y_hat = model(x)

        y_hat = y_hat.detach().numpy()

        ap = utils.average_precision_3(y_hat,y.numpy(),-1)
        break


def nal_eval_metrics():
    classes= [11,12]
    dataset = SHAPESDATASET(cache=False).get_SHAPES_dataset(split="test")
    
    aba_path = ""

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for (img_path, _) in dataset[classes[0]]:

        if not os.path.isfile(img_path):
            continue
    
        prediction = shapes.shapes_9_nal_inference(img_path,aba_path,include_pos=True)

        if prediction:
            tp+=1
        else:
            fn+=1
        

    for (img_path, _) in dataset[classes[1]]:

        if not os.path.isfile(img_path):
            continue
    
        prediction = shapes.shapes_9_nal_inference(img_path,aba_path,include_pos=True)

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

def evaluate_classification(y_true, y_pred, class_names, img_name="confusion_matrix.png"):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    cm = confusion_matrix(y_true, y_pred)
    
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(img_name)


def to_numpy(x):
    return x.cpu().detach().numpy()

def renormalize(x):
    return x / 2. + 0.5  # [-1, 1] to [0, 1]


def test_clustering():
    transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])
    dataset = SHAPESDATASET(transform=transform)
    loader = DataLoader(dataset,32,num_workers=2)

    nal = shapes.get_shape_9_nal_model()
    model= nal.model


    for t, (x, y) in enumerate(loader):
        x = (x / 127.5 ) - 1
        y = y.float()
        _ , _ ,_ ,slot, _ = model(x)
        c = slots.cluster_slots(slot)
        break

    plt.figure(figsize=(10, 6))
    sns.heatmap(c.permute(1, 0, 2).squeeze().cpu().numpy(), cmap="viridis", square=True)
    plt.title("Cluster Assignments for All Batches")
    plt.xlabel("Slot Dimensions")
    plt.ylabel("Slots")
    plt.show()

        
if __name__ == "__main__": 

    dataset_test = CLEVRHans(transform=transform,split="test")
    aba_path = "clevr_bk_1_SOLVED.aba"

    y_pred = []
    y_true = []
    
    for i in range(len(dataset_test)):
        image = dataset_test[i]["input"]
        target = dataset_test[i]["class"]

        pred = clevr.clevr_nal_inference(image,aba_path)
        y_pred.append(pred)
        y_true.append(np.argmax(target))

    
