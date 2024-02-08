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

class Justraigs(Dataset):
    
    def __init__(self, dataframe, is_train=True, is_valid=False, is_test=False):
        self.dataframe, self.is_train, self.is_valid = dataframe, is_train, is_valid
        
        # Data Augmentation
        # If training and sample belongs to minority class, add additional augmentation
        if is_train:
            if 'target' in self.dataframe.columns and (self.dataframe['target'][self.dataframe.index] == 1).any():
                # Additional augmentation for minority class (positive)
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop((224,224), scale=(0.4, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                # Augmentation for majority class (negative)
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        else:
            # Validation/Test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        # Select path and read image
        img_name = self.dataframe['Eye ID'][index]
        image_path = f'train_images/{img_name}.jpg'
        image = io.imread(image_path)
        
        # Apply transforms
        image = self.transform(image)

        
        # If train/valid: image + class | If test: only image
        if self.is_train or self.is_valid:
            return (image, self.dataframe['target'][index])
        else:
            return (image)
        
print(resnet34(pretrained=True))

class ResNet34Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define Feature part (IMAGE)
        self.features = resnet34(pretrained=True) # 1000 neurons out
        
        for param in self.features.parameters():
            param.requires_grad = False
  
        # Define Classification part
        self.classification = nn.Linear(1000, 1)
        
        
    def forward(self, image):
        # Image CNN
        image = self.features(image)
        
        # Classifier
        out = self.classification(image)
        
        return out
    
model = ResNet34Network();
model = model.to(device)

print(model)

# Data object and Loader
train_df = pd.read_csv('Rtrain_folds3.csv')
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
        labels_example = torch.tensor(labels, dtype=torch.float32)
        break
        
    print('Data shape:', image_example.shape)
    print('Label:', labels_example)

    learning_rate = 0.0005
    epochs = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initiate the model
    model = model
    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
    # The dataset is imbalance 90:10
    # Apply weight of 9.0 for minority class
    weight_for_minority_class = 7.0     #pos_weight=torch.tensor([weight_for_minority_class])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_for_minority_class]))


    # Create Data instances
    train = Justraigs(train_data, is_train=True, is_valid=False, is_test=False)
    valid = Justraigs(val_data, is_train=False, is_valid=True, is_test=False)

    # Dataloaders
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=16, shuffle=True)

    # === EPOCHS ===
    for epoch in range(epochs):
        print(f'epoch : {epoch} start!')
        start_time = time.time()
        correct = 0
        train_losses = 0

        # === TRAIN ===
        # Sets the module in training mode.
        model.train()

        for images, labels in train_loader:
        # Save them to device
            images = torch.tensor(images, device=device, dtype=torch.float32)
            labels = torch.tensor(labels, device=device, dtype=torch.float32)

            # Clear gradients first; very important, usually done BEFORE prediction
            optimizer.zero_grad()

            # Log Probabilities & Backpropagation
            out = model(images)
            loss = criterion(out, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_losses += loss.item()
            # From log probabilities to actual probabilities
            train_preds = torch.round(torch.sigmoid(out)) # 0 and 1
            # Number of correct predictions
            correct += (train_preds.cpu() == labels.cpu().unsqueeze(1)).sum().item()

        # Compute Train Accuracy
        train_acc = correct*100 /  int(len(train_data))
        print(f'Epoch :{epoch + 1} - train accuracy: {train_acc}')
        
        # === EVAL ===
        model.eval()

        # Create matrix to store evaluation predictions (for accuracy)
        valid_preds = torch.zeros(size = (len(valid), 1), device=device, dtype=torch.float32)


        # Disables gradients (we need to be sure no optimization happens)
        with torch.no_grad():
            for k, (images, labels) in enumerate(valid_loader):
                    images = torch.tensor(images, device=device, dtype=torch.float32)
                    labels = torch.tensor(labels, device=device, dtype=torch.float32)

                    out = model(images)
                    pred = torch.sigmoid(out)
                    valid_preds[k*images.shape[0] : k*images.shape[0] + images.shape[0]] = pred

            # Compute accuracy
            valid_acc = accuracy_score(val_data['target'].values, 
                                                torch.round(valid_preds.cpu()))*100
                
            # Compute ROC
            valid_roc = roc_auc_score(val_data['target'].values, 
                                                valid_preds.cpu())

            # Compute time on Train + Eval
            duration = str(datetime.timedelta(seconds=time.time() - start_time))[:7]


            # PRINT INFO
            print('{} | Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} ROC: {:.3}'.\
                        format(duration, epoch+1, epochs, train_losses, train_acc, valid_acc,valid_roc))
        
        
torch.save(model.state_dict(), './model.pt')

from IPython.display import FileLink
FileLink(r'model.pt')

model_lo = ResNet34Network();
model_lo = model_lo.to(device)
model_lo.load_state_dict(torch.load('model.pt'))
model_lo.eval()


 # Data object and Loader
df = pd.DataFrame({'Eye ID':['TRAIN096227',
        'TRAIN093510',
         'TRAIN048549',
         'TRAIN042768',
         'TRAIN003632',
         'TRAIN047223',
         'TRAIN091681',
         'TRAIN032298',
         'TRAIN071486',
         'TRAIN070623'], 'Final Label':['NRG', 'NRG', 'RG', 'NRG', 'NRG', 'NRG', 'NRG', 'NRG', 'NRG', 'NRG'], 'Fellow Eye ID': ['TRAIN078050',
         'TRAIN054932',
         'TRAIN049146',
         'TRAIN055785',
         'TRAIN069796',
         'TRAIN010172',
         'TRAIN014003',
         'TRAIN089327',
         'TRAIN067065',
         'TRAIN034062'], 'Age':[69.0, 55.0, 67.0, 47.0, 55.0, 55.0, 37.0, 63.0, 44.0, 62.0], 'target':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]})       


dataset = Justraigs(df, is_train=False, is_valid=True, is_test=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Get a sample
for image, label in loader:
    image = torch.tensor(image, device=device, dtype=torch.float32)
    label = torch.tensor(label, device=device, dtype=torch.float32)
    out = model(image)
    loss = criterion(out, label.unsqueeze(1))
    pred = torch.sigmoid(out)

print('loss: ', loss)
print('Label:', label)
print('Pred:', pred)
print('out:', torch.round(pred))

print('__________________________________________________________________')