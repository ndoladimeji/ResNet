# System
import os, os.path
import PIL
from PIL import Image             
import gc
import time
import datetime

# Basics
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm     

# SKlearn
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

from skimage import io

# PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision.models import resnet34
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# For reproducibility
seed = 1234

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)


# Dataset class for the new task
class Justraigs(Dataset):

    def __init__(self, dataframe, is_train=True, is_valid=False, is_test=False):
        self.dataframe, self.is_train, self.is_valid = dataframe, is_train, is_valid
        
        # Define transformations
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Load image
        img_name = self.dataframe['Eye ID'][index]
        image_path = f'train_images/{img_name}.jpg'
        image = io.imread(image_path)
        image = self.transform(image)

        # Additional features for justification of referable glaucoma
        additional_features = self.dataframe.iloc[index, 4:14].values.astype(np.float32)

        # If train/valid: image + features | If test: only image
        if self.is_train or self.is_valid:
            return (image, additional_features)
        else:
            return (image)

print(resnet34(pretrained=True))


# Define the model for the new task
class ResNet34Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = resnet34(pretrained=True)

        for param in self.features.parameters():
            param.requires_grad = False

        self.additional_features_classifier = nn.Linear(1000, 10)

    def forward(self, image):
        image = self.features(image)
        additional_features_prediction = torch.sigmoid(self.additional_features_classifier(image))
        return additional_features_prediction

# Instantiate the new model
model = ResNet34Network()
model = model.to(device)

print(model)

# Data object and Loader
train_df = pd.read_csv('referable_glaucoma.csv')
dataset = Justraigs(train_df, is_train=True, is_valid=False, is_test=False)
loader = DataLoader(dataset, batch_size=20, shuffle=True)

# Load the fold indices from the CSV file
train_folds = train_df

# Iterate over the folds
for fold in range(3):
    # Get the training and validation data for the current fold
    train_data = train_folds[train_folds['kfold'] != fold].reset_index(drop=True)
    val_data = train_folds[train_folds['kfold'] == fold].reset_index(drop=True)

    # Perform the training and validation using the algorithm/model
    # Train the algorithm/model on the training set
    # Validate the algorithm/model on the validation set
    # Adjust the hyperparameters, train the model, and evaluate its performance

    # Get a sample
    for image, labels in loader:
        image_example = image
        additional_features_example = torch.tensor(labels, dtype=torch.float32)
        break
        
    print('Data shape:', image_example.shape)
    print('Label:', additional_features_example)

    learning_rate = 0.0005
    epochs = 1

    # Initiate the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Create Data instances
    train = Justraigs(train_data, is_train=True, is_valid=False, is_test=False)
    valid = Justraigs(val_data, is_train=False, is_valid=True, is_test=False)

    # Dataloaders
    train_loader = DataLoader(train, batch_size=10, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=5, shuffle=True)

    # === EPOCHS ===
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}...')
        start_time = time.time()
        correct = 0
        train_losses = 0

        # === TRAIN ===
        # Sets the module in training mode.
        model.train()
        
        for images, additional_features in train_loader:
            # Save them to device
            images = torch.tensor(images, device=device, dtype=torch.float32)
            additional_features = torch.tensor(additional_features, device=device, dtype=torch.float32)
            
            optimizer.zero_grad()
            additional_features_pred = model(images)
            
            loss = criterion(additional_features_pred, additional_features)
            loss.backward()
            optimizer.step()
    
            train_losses += loss.item()
            # From log probabilities to actual probabilities
            additional_features_pred = torch.round(torch.sigmoid(additional_features_pred)) # 0 and 1
            # Number of correct predictions
            correct += (additional_features_pred.cpu() == additional_features.cpu()).sum().item()

        # Compute Train Accuracy
        train_acc = correct*100 /  int(len(train_data))
        print(f'Epoch :{epoch + 1} - train accuracy: {train_acc}')
        
        # === EVAL ===
        model.eval()

        # Create matrix to store evaluation predictions (for accuracy)
        valid_preds = torch.zeros(size = (len(valid), 10), device=device, dtype=torch.float32)


        # Disables gradients (we need to be sure no optimization happens)
        with torch.no_grad():
            for k, (images, additional_features) in enumerate(valid_loader):
                    out = model(images)
                    pred = torch.sigmoid(out)
                    valid_preds[k*images.shape[0] : k*images.shape[0] + images.shape[0]] = pred

            # Compute accuracy
            valid_acc = accuracy_score(val_data.iloc[:, 4:14].values, 
                                                torch.round(valid_preds.cpu()))*100
                
            # Compute ROC
            valid_roc = roc_auc_score(val_data.iloc[:, 4:14].values, 
                                                valid_preds.cpu())

            # Compute time on Train + Eval
            duration = str(datetime.timedelta(seconds=time.time() - start_time))[:7]


            # PRINT INFO
            print('{} | Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} ROC: {:.3}'.\
                        format(duration, epoch+1, epochs, train_losses, train_acc, valid_acc,
                                valid_roc))
        
# Save the model   
torch.save(model.state_dict(), './modeljust.pt')