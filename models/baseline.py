import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datasets.CLEVR.CLEVR import CLEVRHans
from datasets.SHAPES.SHAPES import SHAPESDATASET
from neuro_modules.evaluation import evaluate_classification


class CustomImageDataset(Dataset):
    def __init__(self, image_label_list, transform=None):
        self.image_label_list = image_label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_label_list)

    def __getitem__(self, idx):
        img_path, label = self.image_label_list[idx]
        image = Image.open(img_path).convert('RGB') 

        if self.transform:
            image = self.transform(image)

        return image, label

class ResNet34Small(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34Small, self).__init__()
        original_model = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.features = output
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer = dict([*self.model.named_modules()])[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class, is_SHAPES=False):
        model_output = self.model(input_image)
        self.model.zero_grad()
        
        if is_SHAPES:
            one_hot_output = torch.zeros(1, 1).to(input_image.device)  
            one_hot_output[0][0] = 1 if target_class == 1 else 0 
        else:
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(input_image.device)
            one_hot_output[0][target_class] = 1


        model_output.backward(gradient=one_hot_output)
        gradients = self.gradients.cpu().data.numpy()[0]
        features = self.features.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(features.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * features[i]

        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        return cam


def show_cam_on_image(img, mask):

    heatmap_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    heatmap_normalized = np.float32(heatmap_colored) / 255
    
    cam = heatmap_normalized * 0.5 + np.float32(img) * 0.5 
    
    cam = cam / np.max(cam)
    
    return np.uint8(255 * cam)

def run_epoch(model: nn.Module, dataloader: DataLoader, optimizer):
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    
    for data in dataloader:
        inputs = data["input"].float().cuda()
        labels = data["class"].float().cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if optimizer != None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
 
    return running_loss

def baseline_shapes(classes, train=False):

    """
        ResNet Baseline used to train binary tasks on SHAPES 
        Args:
            classes: list of class ids to train shapes
            train: boolean to train resnet model
    """

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64), antialias=None),
            transforms.PILToTensor(), 
        ])

    
    dataset = SHAPESDATASET().get_SHAPES_dataset()
    processed_data = []
    classes = classes
    for i in range(len(classes)):
        n = len(dataset[classes[i]])
        for j in range(n):
            processed_data.append((dataset[classes[i]][j][0],i))

    
    dataset = CustomImageDataset(processed_data, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = ResNet34Small(num_classes=1).cuda()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if train:
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in data_loader:
                inputs = inputs.float().cuda()
                labels = labels.float().cuda().unsqueeze(0)
                optimizer.zero_grad()
                outputs = model(inputs).T
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch {epoch+1}, Loss: {running_loss/len(data_loader)}')

        torch.save(model.state_dict(), 'resnet_model_shapes.pt')

    model.load_state_dict(torch.load('resnet_model_shapes.pt'))
    model.eval() 

    grad_cam = GradCAM(model, target_layer='features.6.5.conv2')
    test_data = CLEVRHans(split="test",transform=transform)


    test_dataset = SHAPESDATASET().get_SHAPES_dataset("test")
    test_data = test_dataset[1][2][0]   # sample image to visualise
    processed_data = []
    
    for i in range(len(classes)):
        n = len(test_dataset[classes[i]])
        for j in range(n):
            processed_data.append((test_dataset[classes[i]][j][0],i))

    
    test_dataset = CustomImageDataset(processed_data, transform=transform)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    predict = []
    true = []

    for inputs, labels in test_data_loader:
        inputs = inputs.float().cuda()
        outputs = model(inputs)
        probabilities = torch.sigmoid(outputs)
    
        predictions = (probabilities >= 0.5).int()
        predict.append(predictions.detach().item())
        true.append(labels.item())

    evaluate_classification(true,predict,["Negative","Postive"])

    image = Image.open(test_data).convert('RGB') 
    image = transform(image).float().cuda().unsqueeze(0)

    outputs = model(image)
    probabilities = torch.sigmoid(outputs)

    predicted_class = (probabilities >= 0.5).int()

    cam_mask = grad_cam.generate_cam(image, predicted_class.item())

    # Load original image
    original_image = Image.open(test_data).convert('RGB')
    original_image = np.array(original_image) / 255

    # Visualize CAM
    cam_image = show_cam_on_image(original_image, cam_mask)
    plt.imshow(cam_image)
    plt.axis("off")
    plt.savefig("shapes_gradcam.png")


def baseline_clevr(train=False):

    """
        ResNet Baseline used to train on Clevr-Hans
        Args:
            train: boolean to train resnet model
    """

    transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])
    
    
    train_dataset = CLEVRHans(split="train", transform=transform,cache=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = CLEVRHans(split="val", transform=transform,cache=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ResNet34Small(num_classes=3)
    model_path = os.getcwd() + "/models/clevr_b1.pt"

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    if train:
        best_loss = 1e6
        num_epochs = 100
        model.train()
        
        log_file_path = "training_log_baseline.txt"

        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as file:
                file.write("Log file created\n")
                print("Log file created at:", log_file_path)
        else:
            print("Log file already exists at:", log_file_path)
        logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        for epoch in range(num_epochs):
            
            train_loss = run_epoch(model,train_loader,optimizer)
           
            if epoch % 4 == 0:
                valid_loss = run_epoch(model,val_loader, None)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    step = int(epoch * len(train_loader))
                    
                    torch.save(model.state_dict(), f"./checkpoints/resnet_baseline/{step}_ckpt.pt")

            logging.info("Epoch {}: Train loss: {:.6f}, Valid loss: {:.6f}".format(epoch, train_loss, valid_loss))
        else:
            logging.info("Epoch {}: Train loss: {:.6f}".format(epoch, train_loss))

    else:

        grad_cam = GradCAM(model, target_layer='features.6.5.conv2')
        test_data = CLEVRHans(split="test",transform=transform)

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval() 

        predict = []
        true = []

        for i in range (len(test_data)):
            with torch.no_grad():
                image = test_data[i]["input"].float().unsqueeze(0)
                outputs = model(image)
                predicted_class = torch.argmax(outputs)
            

            predict.append(predicted_class.item())
            true.append(np.argmax(test_data[i]["class"]))


        true = np.array(true)
        predict = np.array(predict)

        evaluate_classification(true,predict,["Class 1", "Class 2", "Class 3"],img_name="cm_baseline.png")

        cam_mask = grad_cam.generate_cam(image, predicted_class.item())

        # Load original image
        original_image = Image.open(test_data[2]["path"]).convert('RGB')
        original_image = np.array(original_image) / 255

        # Visualize CAM
        cam_image = show_cam_on_image(original_image, cam_mask)
        plt.imshow(cam_image)
        plt.axis('off')
        plt.savefig("clevr_gradcam.png")


if __name__ == '__main__':
    ## SHAPES Classes: [1,2] [3,4] [5,6] [7,8] [9,10] [11,12]
    class_id = [1,2]
    baseline_shapes(class_id,train=True)
    baseline_clevr(train=True)