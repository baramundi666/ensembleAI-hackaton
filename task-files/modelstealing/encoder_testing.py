import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from taskdataset import TaskDataset

from autoencoder import Autoencoder
# Wczytaj model autoenkodera z pliku
# autoencoder = torch.load('models/autoencoder_model.pt')
# Load the state dict
state_dict = torch.load('image_to_vector_model.pt')

# Extract the encoder state dict
# encoder_state_dict = {k: v for k, v in state_dict.items() if k.startswith('encoder')}

# Load the encoder state dict into the autoencoder model
autoencoder = Autoencoder()
# autoencoder.encoder.load_state_dict(encoder_state_dict)

# Set the model to evaluation mode
autoencoder.eval()

# autoencoder = Autoencoder()
# autoencoder.load_state_dict(torch.load('image_to_vector_model.pt'))
# autoencoder.eval()
# Wczytaj przykładowy obraz

dataset = torch.load("data/ModelStealingPub.pt")
sample_image = dataset.imgs[0]

# Przygotuj obraz do przekazania przez model
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Wymagane wymiary obrazu przez autoenkoder
    transforms.ToTensor()
])
input_image = transform(sample_image).unsqueeze(0)

# Przeprowadź inferencję na przykładowym obrazie
with torch.no_grad():
    reconstructed_image = autoencoder(input_image)
    print(reconstructed_image)
# # Wyświetl oryginalny obraz i odtworzoną wersję
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Oryginalny obraz')
# plt.imshow(sample_image)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Odtworzony obraz')
# plt.imshow(reconstructed_image.squeeze(0).permute(1, 2, 0))
# plt.axis('off')

# plt.show()
