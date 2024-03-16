import requests
def model_stealing_submission(path_to_onnx_file: str):
    SERVER_URL = "[paste server url here]"
    ENDPOINT = "/modelstealing/submit"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "3FNWQO9kLVQmnLj4"

    with open(path_to_onnx_file, "rb") as onnx_file:
        response = requests.post(
            URL, files={"file": onnx_file}, headers={"token": TEAM_TOKEN}
        )

        if response.status_code == 200:
            return response.content["score"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")