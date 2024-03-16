
from taskdataset import TaskDataset
import torch
from torchvision.models import resnet18
from torchvision.transforms import transforms

dataset = torch.load("modelstealing/data/ModelStealingPub.pt")
imgs = dataset.imgs

model = resnet18(pretrained=False)  
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 13000)  

model.load_state_dict(torch.load('wytrenowany_model.pt'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

imgs_tensor = torch.stack([transform(img.convert("RGB")) for img in imgs[:100]])

with torch.no_grad():
    predictions = model(imgs_tensor)

predicted_classes = torch.argmax(predictions, dim=1)

print("Przewidywane klasy:")
print(predicted_classes)
