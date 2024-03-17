from taskdataset import TaskDataset
import torch 
import requests
import json
import pandas as pd
from dataset_augmentation import augment_dataset, augment_dataset_other

from PIL import Image
def model_stealing(path_to_png_file: str):
    SERVER_URL = "http://34.71.138.79:9090"
    ENDPOINT = "/modelstealing"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "3FNWQO9kLVQmnLj4"

    with open(path_to_png_file, "rb") as img_file:
        response = requests.get(
            URL, files={"file": img_file}, headers={"token": TEAM_TOKEN}
        )

        if response.status_code == 200:
            return json.loads(response.content.decode())["representation"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")

def genearte_png():
    dataset = torch.load("data/ModelStealingPub.pt")   
    dataset.imgs = dataset.imgs[:300]
    dataset.labels = dataset.labels[:300]
    dataset.ids = dataset.ids[:300]    
    dataset = augment_dataset(dataset)

    for i in range(len(dataset.imgs)):
        dataset.imgs[i].save(f'png_files/photo+{i}.png')

def send_queries():
    for i in range(0,900):
        new_value = {"ID": i, "model_output": model_stealing(f"png_files/photo+{i}.png")}
        df = pd.DataFrame([new_value])  # Tworzymy tymczasowy DataFrame dla nowej wartości
        df.to_csv("outputs/output2.csv", sep=';', mode='a', index=False, header=not i)  # Dopisujemy do pliku CSV
        print(i)

send_queries()
# 




