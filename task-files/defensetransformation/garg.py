import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import shuffle
import json

def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    url = "http://34.71.138.79:9090" + endpoint
    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": "3FNWQO9kLVQmnLj4"})
        if response.status_code == 200:
            return json.loads(response.content.decode())["representation"]
        else:
            raise Exception(
                f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
            )

if __name__ == "__main__":
    data = np.load(
        "defensetransformation/data/DefenseTransformationEvaluate.npz"
    )
    print(data["labels"], data["representations"].shape)

    # data = np.load("defensetransformation/data/DefenseTransformationSubmit.npz")
    # print(data["representations"].shape)

    data = data["representations"]
    desired_size = np.random.randint(32, 192)

    # Perform dimensionality reduction using Randomized PCA
    pca = PCA(n_components=desired_size, random_state=42)
    transformed_data = pca.fit_transform(data)

    # Quantize the transformed representations
    quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
    transformed_data = quantile_transformer.fit_transform(transformed_data)

    # Additive Gaussian Noise
    noise = np.random.normal(0, 0.1, transformed_data.shape)
    transformed_data += noise

    # Feature Shuffling
    transformed_data = shuffle(transformed_data, random_state=42)
    #np.savez_compressed('defensetransformation/file.npz', t=transformed_data)
    defense_submit("defensetranformation/file/npz")

    # Save the transformed representations
    # np.savez_compressed('protected_representations.npz', representations=transformed_data)