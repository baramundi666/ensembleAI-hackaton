import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from taskdataset import TaskDataset
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
from imperator import ImageToVector

model = ImageToVector()
model.load_state_dict(torch.load('image_to_vector_model.pt'))
model.eval()


# Preprocess the sample image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = torch.load("data/ModelStealingPub.pt")
imgs = dataset.imgs[1:66]

imgs_tensor = torch.stack([transform(img.convert("RGB")) for img in imgs])  # Add batch dimension

# Perform inference to obtain the vector representation
with torch.no_grad():
    output_vectors = model(imgs_tensor)

print(output_vectors.shape)