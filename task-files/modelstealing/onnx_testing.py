import onnxruntime
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
from taskdataset import TaskDataset

# Wczytaj obraz
dataset = torch.load("data/ModelStealingPub.pt")
sample_image = dataset.imgs[0]

# Przygotuj obraz do przekazania przez model
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Wymagane wymiary obrazu przez model ONNX
    transforms.ToTensor()
])
input_image = transform(sample_image).unsqueeze(0).numpy()  # Przekształć na tablicę NumPy

# Wczytaj model ONNX
onnx_model_path = 'model.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Przeprowadź inferencję na przykładowym obrazie
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
output = ort_session.run([output_name], {input_name: input_image})[0]
print(output)
# Przekształć wynik z tablicy NumPy na obraz PIL
reconstructed_image = transforms.ToPILImage()(torch.tensor(output[0]))
# Wyświetl oryginalny obraz i odtworzoną wersję
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Oryginalny obraz')
plt.imshow(sample_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Odtworzony obraz')
plt.imshow(reconstructed_image)
plt.axis('off')

plt.show()
