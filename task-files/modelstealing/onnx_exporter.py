import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.onnx

# Wczytaj model autoenkodera
from autoencoder import Autoencoder
model = Autoencoder()
model.load_state_dict(torch.load('models/autoencoder_model.pt'))
model.eval()

# Przygotuj przykładowe dane wejściowe
input_example = torch.randn(1, 3, 32, 32)  # Przykładowe dane wejściowe (1 przykład, 3 kanały, 32x32 rozmiar obrazu)

# Eksportuj model do formatu ONNX
torch.onnx.export(model, input_example, 'model.onnx', export_params=True)
