from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.dnn import SimpleNet, Trainer, MLP_dropout
from datasets.buildings_dataset import Buildings
from active_learning.active_learn import ActiveLearning

## Input data
dataset_th_file = "datasets/subset_build_6kB_dataset.pth"

num_train = 30
num_test = 1500

batch_size = 32
learning_rate = 0.0001
num_epochs = 200
hidden_size = 100
layers = 2
num_classes = 2
dropout_rate = 0.1
# num_classes = 18

num_active_iter = 1000
num_forwards = 100


# Assuming you have created a Buildings dataset instance named buildings_dataset
buildings_dataset = torch.load(dataset_th_file)

# Calculate the length of the dataset
total_samples = len(buildings_dataset)
print("Total number of samples in the dataset: ", total_samples)

num_pool = total_samples - num_train - num_test

# Generate random indices for the downscaled dataset
# Ensure reproducibility with a fixed seed, if necessary
# np.random.seed(42)  # Uncomment to make the selection reproducible
# indices = np.random.choice(len(buildings_dataset), downscaled_size, replace=False)

# Use random_split to split the dataset into training and testing sets
train_ds, test_ds, pool_ds = random_split(buildings_dataset, [num_train, num_test, num_pool])

print("Number of samples in the training set: ", len(train_ds))
print("Number of samples in the testing set: ", len(test_ds))
print("Number of samples in the pool set: ", len(pool_ds))

### To reset the datasets
# train_ds_0 = train_ds
# test_ds_0 = test_ds
# pool_ds_0 = pool_ds

test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

f1_score_AL = []
active_learn = ActiveLearning(num_active_points=1)

for i in range(num_active_iter):
    print("Active learning iteration: ", i)
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    pool_loader = DataLoader(pool_ds, batch_size=len(pool_ds), shuffle=False)


    # Instantiate the classifier
    input_size = len(train_ds[0][0]) # It should be stored in logs
    input_size = 384 # Only considering aerial images
    net = MLP_dropout(input_size, hidden_size, layers, num_classes, dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    trainer = Trainer(net, train_loader, test_loader, criterion, optimizer, num_epochs, patience=400)

    trainer.train()
    f1_score_AL.append(trainer.f1_score)
    
    ## Loop
    idx_pool = pool_ds.indices
    idx_train = train_ds.indices

    trainer.model.load_state_dict(trainer.best_model)

    ## Get active points
    selected_idx_pool = active_learn.get_active_points(trainer.model, num_forwards, buildings_dataset, idx_pool)
    
    ## Updated indices based on selected samples
    idx_pool_ = [idx for idx in idx_pool if idx not in selected_idx_pool]
    idx_train_ = idx_train + selected_idx_pool

    ## Updated subdatasets based on selected samples
    train_ds = torch.utils.data.dataset.Subset(buildings_dataset, idx_train_) 
    pool_ds = torch.utils.data.dataset.Subset(buildings_dataset, idx_pool_) 

    print(len(pool_ds), len(train_ds))