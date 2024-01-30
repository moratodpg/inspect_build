# import os
# from datetime import datetime
# import pickle
# import argparse
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dnn import SimpleNet, Trainer, MLP_dropout
from datasets.buildings_dataset import Buildings
## Input data
dataset_th_file = "datasets/subset_build_6kB_dataset.pth"

batch_size = 64
train_ratio = 0.8
learning_rate = 0.0001
num_epochs = 50
hidden_size = 200
num_classes = 2
# num_classes = 18

# Assuming you have created a Buildings dataset instance named buildings_dataset
buildings_dataset = torch.load(dataset_th_file)

# Calculate the length of the dataset
total_samples = len(buildings_dataset)
print("Total number of samples in the dataset: ", total_samples)

# Calculate the number of samples for training and testing
num_train = int(train_ratio * total_samples)
num_test = total_samples - num_train

# Use random_split to split the dataset into training and testing sets
train_dataset, test_dataset = random_split(buildings_dataset, [num_train, num_test])
## In active learning: train_dataset will include new buildings, while test_dataset will have less buildings

# Create DataLoader for training set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Create DataLoader for testing set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the classifier
input_size = len(train_dataset[0][0])

# net = SimpleNet(input_size, hidden_size, num_classes)
net = MLP_dropout(input_size, hidden_size, 2, num_classes, 0.1)


# Define loss function and optimizer
# criterion = FocalLoss()
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.499674613973851]))
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.15385385567610865, 0.8461461443238913]))
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.8461461443238913, 0.15385385567610865]))
# criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Instantiate TrainerTester
trainer = Trainer(net, train_loader, test_loader, criterion, optimizer, num_epochs)
# trainer = MultiClassTrainer(net, train_loader, test_loader, criterion, optimizer, num_epochs, num_classes)

# Train and test the model
trainer.train()