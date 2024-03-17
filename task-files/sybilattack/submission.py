import os
import numpy as np
import requests
from typing import List
from load_dataset import load
from typing import Tuple
import torch
from taskdataset import TaskDataset

def sybil_attack_reset():
    SERVER_URL = "http://34.71.138.79:9090"
    ENDPOINT = "/sybil/reset"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "3FNWQO9kLVQmnLj4"

    response = requests.post(
        URL, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        print("Endpoint rested successfully")
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")

def sybil_attack(ids: List[int], home_or_defense: str, binary_or_affine: str):
    if home_or_defense not in ["home", "defense"] or binary_or_affine not in ["binary", "affine"]:
        raise "Invalid endpoint"
    
    SERVER_URL = "http://34.71.138.79:9090"
    ENDPOINT = f"/sybil/{binary_or_affine}/{home_or_defense}"
    URL = SERVER_URL + ENDPOINT
    
    TEAM_TOKEN = "3FNWQO9kLVQmnLj4"
    ids = ids = ",".join(map(str, ids))

    response = requests.get(
        URL, params={"ids": ids}, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        return response.content["representations"]
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")
    
def transform(ids,imgs,labels,transform, index) -> Tuple[int, torch.Tensor, int]:
    id_ = ids[index]
    img = imgs[index]
    if not transform is None:
        img = transform(img)
    label = labels[index]
    return id_, img, label

if __name__ == "__main__":
    dataset = load()
    # print(dataset.ids, dataset.imgs, dataset.labels)
    np.savez(
        "data/submission.npz",
        ids=[i for i in range(2*10**5)],
        representations=np.matrix([[i for i in range(2*10**5)] for j in range(192)])
    )
    ids = [i for i in range(2*10**5)]
    sybil_attack(ids, "defense", "affine")
    #sybil_attack_reset()
