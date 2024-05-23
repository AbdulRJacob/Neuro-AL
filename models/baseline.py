import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datasets.CLEVR.CLEVR import CLEVRHans


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

    def generate_cam(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
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
        cam = cam / cam.max()  # Normalize

        return cam


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)



if __name__ == "__main__":


    transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])
    
    
    train_dataset = CLEVRHans(split="train", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = CLEVRHans(split="val", transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ResNet34Small(num_classes=3)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop (simplified)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs = data["input"].float()
            labels = data["class"].float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    
    grad_cam = GradCAM(model, target_layer='layer4')
    test_data = CLEVRHans(split="test",transform=transform)
    img = test_data[1]
    # Forward pass to get the class prediction
    model.eval()
    outputs = model(img)
    _, predicted_class = outputs.max(1)

    cam_mask = grad_cam.generate_cam(img, predicted_class.item())

    # Load original image
    original_image = img.convert('RGB')
    original_image = np.array(original_image) / 255

    # Visualize CAM
    cam_image = show_cam_on_image(original_image, cam_mask)
    plt.imshow(cam_image)
    plt.show()