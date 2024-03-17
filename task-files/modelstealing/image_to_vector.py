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

class ImageToVector(nn.Module):
    def __init__(self):
        super(ImageToVector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Linear layer to map flattened output to a vector of desired size
        self.fc = nn.Linear(64 * 4 * 4, 512)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        return x