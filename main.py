# import necessary libriaries
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize
from L20_dataloader2 import MyDataLoader
import torchvision
import os
import cv2
from datetime import datetime
import numpy as np

# Determine devises to use (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define hyperparameters
batch_size = 16
epochs = 40
learning_rate = 0.0005  # Adjusted learning rate

# Define the custom model architecture 
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),  # Change 1 to 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU()
        )
        #self.fusion = nn.Conv2d(256, 256, 1, padding=0)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid(),
        )
        #self.customized = MyLayer()
        #[0,1] = [0,256] - 128

    def forward(self, x):
        # x = gets the input image
        encoder_features = self.encoder(x)
        x = self.decoder(encoder_features)
        return x

# defining training dataset
path_train_gray = ''
path_train_rgb = ''
# defining testing data
path_test_gray = ''
path_test_rgb = ''
alist = os.listdir()

# Define data transformations for input normalization 
data_transform = nn.Sequential(
    torchvision.transforms.ConvertImageDtype(torch.float32),
    #nn.Sigmoid()
    #Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
)

# Define data loading
def data_loading(path_gray, path_rgb, data_transform):
    dataset = MyDataLoader(path_gray, path_rgb, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

# define data_loaders for training and testing data
train_dataloader = data_loading(path_train_gray, path_train_rgb, data_transform)
test_dataloader = data_loading(path_test_gray, path_test_rgb, data_transform)

# Initialize the custom model and move it to the selected device 
model = MyModel().to(device)

# Define loss function
loss_fn = nn.MSELoss()

# Define optimizer
# adamW is faster optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    len_dataloader = len(dataloader)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Batch [{batch_idx}/{len_dataloader}]\tLoss: {loss.item():.4f}')


# Testing loop
def test_loop(dataloader, model):
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if batch_idx == 1:
                output_image = outputs[0].cpu().numpy().transpose(1, 2, 0)
                #output_image = (output_image + 1) / 2.0  # De-normalize to [0, 1]
                output_image = (output_image * 255).astype(np.uint8)
                cv2.imwrite(f'/{datetime.now()}.png', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
                
                targets = targets[0].cpu().numpy().transpose(1, 2, 0)
                targets = (targets * 255).astype(np.uint8)
                cv2.imwrite(f'/{datetime.now()}.png',cv2.cvtColor(targets, cv2.COLOR_RGB2BGR))
                
# Training and testing loop
for epoch in range(epochs):
    print(f'Epoch [{epoch}/{epochs}]')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model)
