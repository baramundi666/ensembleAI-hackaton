from taskdataset import TaskDataset
import torch
from torchvision.models import resnet18
from torchvision.transforms import transforms

TRANING_SET_SIZE = 39

def map_labels(labels):
    d = dict()
    counter = 1
    for label in labels:
        if label not in d:
            d[label] = counter
            counter += 1
    return d

dataset = torch.load("data/ModelStealingPub.pt")
imgs = dataset.imgs

model = resnet18(pretrained=False)  
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, TRANING_SET_SIZE)  

model.load_state_dict(torch.load('wytrenowany_model.pt'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

imgs_tensor = torch.stack([transform(img.convert("RGB")) for img in imgs])

with torch.no_grad():
    predictions = model(imgs_tensor)

predicted_classes = torch.argmax(predictions, dim=1)

labels = dataset.labels
labels_dict = map_labels(labels)
score = 0 


for i in range(imgs_tensor.size(0)):
    if labels_dict[labels[i]] == predicted_classes[i]:
        score += 1

print("Dokładność modelu: ", score/imgs_tensor.size(0))