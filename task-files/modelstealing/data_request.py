from taskdataset import TaskDataset
import torch 
import requests
import json
import pandas as pd

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

df = pd.DataFrame(columns=["ID", "model_output"])

for i in range(1, 13000, 200):
    new_value = {"ID": i, "model_output": model_stealing(f"png_photos/img{i}.png")}
    df = df.append(new_value, ignore_index=True)

df.to_csv("outputs/output.csv", sep=';', index=False)



