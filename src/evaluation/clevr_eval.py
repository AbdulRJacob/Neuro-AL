import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from sklearn.metrics import adjusted_rand_score

from data.CLEVR import CLEVRHans, CLEVR

import utils.utils as utils
import training.clevr_train_symbolic as clevr
from data.CLEVR import dataset as ds

import inference.clevr_inference as clevr_inf

transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])


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

def calcualte_AP():
    dataset = CLEVR(transform=transform)
    loader = DataLoader(dataset,128,num_workers=2)

    nal = clevr.get_clevr_nal_model()
    model= nal.model

    for ldr in loader:
        x = ldr[0]
        y = ldr[1]

        x = (x / 127.5 ) - 1
        y = y.float()
        _ , _ ,_ ,_, y_hat = model(x)

        y_hat = y_hat.detach().numpy()

        ap = utils.average_precision(y_hat,y.numpy(),-1)
        break

    print("Average Precision: " + str(ap))



if __name__ == "__main__": 

    with open("config/clevr_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    class_order = config["sym_training"]["class_order"]
    pass_1_framework = config["inference"]["first_pass"]
    pass_2_framework = config["inference"]["second_pass"]


    print("========== CLEVR ML METRICS ===============")
    dataset_test = CLEVRHans(split="test")
    aba_path = [pass_1_framework,pass_2_framework]

    y_pred = []
    y_true = []
    
    for i in range(len(dataset_test)):
        image = dataset_test[i]["input"]
        target = dataset_test[i]["class"]

        pred = clevr_inf.clevr_nal_inference(image,aba_path,class_order,isPath=False)
        y_pred.append(pred)
        y_true.append(np.argmax(target))

    evaluate_classification(y_true,y_pred,["Class 1", "Class 2", "Class 3"], "clevr_cm.png")

    print("========= AP AND ARI METRICS ============")
    calcualte_AP()  
    ari_clevr()
