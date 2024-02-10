import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
import torch.nn as nn
import pandas as pd
import numpy as np
from skimage import io

import warnings
warnings.filterwarnings("ignore")

# Define the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained binary classifier model
class ResNet34NetworkBinary(nn.Module):
    def __init__(self):
        super(ResNet34NetworkBinary, self).__init__()
        self.features = resnet34(pretrained=True)
        for param in self.features.parameters():
            param.requires_grad = False
        self.classification = nn.Linear(1000, 1)

    def forward(self, image):
        image = self.features(image)
        out = self.classification(image)
        return torch.sigmoid(out)

binary_model = ResNet34NetworkBinary()
binary_model.load_state_dict(torch.load('model.pt'))
binary_model = binary_model.to(device)
binary_model.eval()

# Load the trained multi-label classifier model
class ResNet34NetworkMultiLabel(nn.Module):
    def __init__(self):
        super(ResNet34NetworkMultiLabel, self).__init__()
        self.features = resnet34(pretrained=True)
        for param in self.features.parameters():
            param.requires_grad = False
        self.classification = nn.Linear(1000, 10)

    def forward(self, image):
        image = self.features(image)
        out = self.classification(image)
        return torch.round(torch.sigmoid(out))

multi_label_model = ResNet34NetworkMultiLabel()
multi_label_model.load_state_dict(torch.load('modeljust.pt'))
multi_label_model = multi_label_model.to(device)
multi_label_model.eval()

# Dataset class for inference
class JustraigsInference(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_name = self.dataframe['Eye ID'][index]
        image_path = f'input/{img_name}'  # Assume image name without extension

        # Check if the image is a stacked TIFF
        if os.path.exists(f'{image_path}.tiff'):
            image = io.imread(f'{image_path}.tiff')
            if len(image.shape) > 2:  # Check if image is stacked
                # Destack the image
                image_list = [self.transform(slice) for slice in image]
                return image_list, img_name
            else:
                image = self.transform(image)
                return image, img_name
        # Check if the image is a single .mha image
        elif os.path.exists(f'{image_path}.mha'):
            # Load .mha image using appropriate library (e.g., SimpleITK)
            # Assuming you have SimpleITK installed, you can replace the following line with the appropriate code to load .mha image
            import SimpleITK as sitk
            image = sitk.ReadImage(f'{image_path}.mha')
            # Convert .mha image to numpy array
            image = sitk.GetArrayFromImage(image)
            image = self.transform(image)
            return image, img_name
        else:
            raise FileNotFoundError(f"Image file not found for {img_name}")

# Transformation for image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the data for inference
df = pd.read_csv('Jtrain_folds.csv')
all_dataset = JustraigsInference(df, transform)
all_data_loader = DataLoader(all_dataset, batch_size=1, shuffle=False)

referable_glaucoma_df = pd.read_csv('referable_glaucoma.csv')
inference_dataset = JustraigsInference(referable_glaucoma_df, transform)
inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

# Store predictions along with image IDs
binary_probabilities = []
binary_predictions = []
multi_label_predictions = []

# Iterate over the images and make predictions for binary classification
for images, image_ids in all_data_loader:
    images = images.to(device)
    
    # Binary classification inference
    with torch.no_grad():
        out = binary_model(images)
        pred = torch.sigmoid(out)

        # Convert rounded predictions to True and False
        rounded_predictions = (torch.round(pred) == 1).bool()

        # Append to lists
        binary_probabilities.extend(pred.cpu().numpy().tolist())
        binary_predictions.extend(rounded_predictions.cpu().numpy().tolist())

# Multi-label classification inference
for images, image_ids in inference_loader:
    images = images.to(device)
    
    with torch.no_grad():
        multi_label_pred = multi_label_model(images)
        multi_label_predictions.extend(multi_label_pred.cpu().numpy().tolist())

# Save binary classification probabilities to a JSON file
with open('test/output/likelihoods.json', 'w') as file:
    json.dump(binary_probabilities, file)

# Round binary predictions and save to JSON file
with open('test/output/binary_decisions.json', 'w') as file:
    json.dump(binary_predictions, file)

# Save multi-label predictions to JSON file
with open('test/output/multi_label_predictions.json', 'w') as file:
    json.dump(multi_label_predictions, file)