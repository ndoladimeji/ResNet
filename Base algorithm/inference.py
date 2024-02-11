import torch
from torchvision import transforms
from torchvision.models import resnet34
import torch.nn as nn
from skimage import io

import warnings
warnings.filterwarnings("ignore")

import numpy
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks


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
        result = (torch.round(torch.sigmoid(out)) == 1).bool()
        return result

multi_label_model = ResNet34NetworkMultiLabel()
multi_label_model.load_state_dict(torch.load('modeljust.pt'))
multi_label_model = multi_label_model.to(device)
multi_label_model.eval()

# Transformation for image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def run():
    _show_torch_cuda_info()

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Binary classification inference
        image = io.imread(jpg_image_file_name)
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = binary_model(image)
            pred = torch.sigmoid(out)

            # Convert rounded predictions to True and False
            is_referable_glaucoma = (torch.round(pred) == 1).bool().numpy()
            is_referable_glaucoma_likelihood = pred.numpy()
        
        # Multi-label classification inference
        if is_referable_glaucoma:
               with torch.no_grad():
                   multi_label_pred = multi_label_model(image)
                   multi_label_pred = multi_label_pred.numpy()
                   
                   features = {k: v for k, v in DEFAULT_GLAUCOMATOUS_FEATURES.items()}
                   for i, pred in enumerate(multi_label_pred):
                       features[list(features.keys())[i]] = pred
        else:
            features = None

        print(f"Running inference on {jpg_image_file_name}")
        ...

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
    return 0


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())