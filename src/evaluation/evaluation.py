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
from data.CLEVR import CLEVRHans, CLEVR
from data.CLEVR import dataset as ds

from models.NAL import NAL
import utils.utils as utils
import training.shapes_train_neural as shapes
import training.clevr_train_neural as clevr
import models.slot_ae as slots

transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])


@torch.no_grad()
def visualise_slots(model, img_path, num_slots, idx=0):
    img_path = NAL.preprocess_image(None,img_path,True)
    recon_combined, recons, masks, slots  = model(img_path)
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

    plt.savefig('slot_vis_test.png')

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


def visualise_slots_tsne(self, p_eg: list[int], n_eg: list[int], num_examples):
    data = self.dataset.get_all_data()
    data_s = []
    rep_negative = []
    for (idx, info) in data:
        _, _, _, slots, _ = self.model(info["input"].unsqueeze(0).float())
        combined_slots = utils.aggregate_slots(slots.detach().numpy(),method="average")
        if np.argmax(info["class"]) == 2:
            data_s.append((idx, str(np.argmax(info["class"])), combined_slots))

    utils.visualize_tsne(data_s,num_examples)


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


def eval_clevr_peception():
    dataset = CLEVR(transform=transform, split="val")
    loader = DataLoader(dataset,1,num_workers=2)
 
    model = clevr.get_clevr_nal_model().model
 
    # Initialize counters for each attribute
    counters = {
        'shape': {'TP': 0, 'FP': 0, 'FN': 0},
        'color': {'TP': 0, 'FP': 0, 'FN': 0},
        'object_size': {'TP': 0, 'FP': 0, 'FN': 0},
        'material': {'TP': 0, 'FP': 0, 'FN': 0},
        'total': 0
    }
 
 
    for _, (x, y) in enumerate(loader):
        x = (x / 127.5 ) - 1
        y = y.float()
        _ , _ ,_ , _ , output = model(x)
 
        cost_matrix = utils.calculate_distances(y,output,11)
        h_match = utils.hungarian_algorithm(cost_matrix)[1]
 
        actual_indices = h_match[0, 0].tolist()
        predicted_indices = h_match[0, 1].tolist()
 
        y = y.squeeze(0)
        output = output.squeeze(0)
 
        matched_actual_labels = [y[i] for i in actual_indices]
        matched_predicted_labels = [output[i] for i in predicted_indices]
 
 
        for actual, predicted in zip(matched_actual_labels, matched_predicted_labels):
            (pred_coords, pred_shape, pred_color, pred_object_size, pred_material,
                _) = process_targets_clevr(predicted.detach().numpy())
           
            (target_coords, target_shape, target_color, target_object_size,
                target_material, target_real_obj) = process_targets_clevr(actual.detach().numpy())
           
            if target_real_obj < 0.5:
                continue
 
            counters['total'] += 1
 
            # Update counters for shape
            if pred_shape == target_shape:
                counters['shape']['TP'] += 1
            else:
                counters['shape']['FP'] += 1
                counters['shape']['FN'] += 1
 
            # Update counters for color
            if pred_color == target_color:
                counters['color']['TP'] += 1
            else:
                counters['color']['FP'] += 1
                counters['color']['FN'] += 1
 
            # Update counters for object_size
            if pred_object_size == target_object_size:
                counters['object_size']['TP'] += 1
            else:
                counters['object_size']['FP'] += 1
                counters['object_size']['FN'] += 1
 
            # Update counters for material
            if pred_material == target_material:
                counters['material']['TP'] += 1
            else:
                counters['material']['FP'] += 1
                counters['material']['FN'] += 1      
 
    def calculate_metrics(attribute_counters):
        TP = attribute_counters['TP']
        FP = attribute_counters['FP']
        FN = attribute_counters['FN']
       
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = TP / counters['total'] if counters['total'] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
       
        return precision, recall, accuracy, f1
   
 
    for attribute in ['shape', 'color', 'object_size', 'material']:
         precision, recall, accuracy, f1 = calculate_metrics(counters[attribute])
         print(f'{attribute.capitalize()} - Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}, F1 Score: {f1}')


def evaluate_classification(y_true, y_pred, class_names, img_name="confusion_matrix.png", normalize=False, cmap=plt.cm.Blues):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
    
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title='Normalized confusion matrix' if normalize else 'Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(img_name)
    plt.show()


def to_numpy(x):
    return x.cpu().detach().numpy()

def renormalize(x):
    return x / 2. + 0.5  # [-1, 1] to [0, 1]

def process_targets_clevr(target):
    """Unpacks the target into the CLEVR properties."""
    coords = target[:3]
    shape = np.argmax(target[3:6])
    color = np.argmax(target[6:14])
    object_size = np.argmax(target[14:16])
    material = np.argmax(target[16:18])
    real_obj = target[18]
    return coords, shape, color, object_size, material, real_obj


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

    analyse_SHAPES()

    # dataset_test = CLEVRHans(split="test")
    # aba_path = ["results/CLEVR/clevr_bk_1_SOVLED.aba","results/CLEVR/clevr_bk_2_SOVLED.aba"]

    # y_pred = []
    # y_true = []
    
    # for i in range(len(dataset_test)):
    #     image = dataset_test[i]["input"]
    #     target = dataset_test[i]["class"]

    #     pred = clevr.clevr_nal_inference(image,aba_path)
    #     y_pred.append(pred)
    #     y_true.append(np.argmax(target))

    # evaluate_classification(y_true,y_pred,["Class 1", "Class 2", "Class 3"], "cm_best.png")