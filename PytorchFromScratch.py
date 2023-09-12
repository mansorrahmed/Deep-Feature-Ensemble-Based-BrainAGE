# Importing all the essential libraries

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Set the device

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Initializing the required hyperparameters

num_classes = 10
input_size = 784
batch_size = 64
lr = 0.0001
epochs = 3


# Collection of data

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

X_train = torchvision.datasets.MNIST(root='/datasets', train=True, download=True, transform=T)
train_loader = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True)

X_test = torchvision.datasets.MNIST(root='/datasets', train=False, download=True, transform=T)
test_loader = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=True)


# Constructing the model

class neural_network(nn.Module):
    def __init__(self, input_size, num_classes):
        super(neural_network, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train the network

for epoch in range(epochs):
    for batch, (data, target) in enumerate(train_loader):
        # Obtaining the cuda parameters
        data = data.to(device=device)
        target = target.to(device=device)

        # Reshaping to suit our model
        data = data.reshape(data.shape[0], -1)

        # Forward propagation
        score = model(data)
        loss = criterion(score, target)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Check the performance

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        if num_samples == 60000:
            print(f"Train accuracy = "
                  f"{float(num_correct) / float(num_samples) * 100:.2f}")
        else:
            print(f"Test accuracy = "
                  f"{float(num_correct) / float(num_samples) * 100:.2f}")

    model.train()
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

