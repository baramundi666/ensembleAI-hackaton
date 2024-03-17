import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.onnx
from taskdataset import TaskDataset

# Wczytaj model autoenkodera
from imperator import ImageToVector
model = ImageToVector()
model.load_state_dict(torch.load('imperator_model.pt'))
model.eval()

input_example = torch.randn(1, 3, 32, 32)  

# Eksportuj model do formatu ONNX
torch.onnx.export(model, input_example, 'models/imperator2.onnx', export_params=True, input_names=["x"])
