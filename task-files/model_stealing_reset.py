import requests
def model_stealing_reset():
    SERVER_URL = "http://34.71.138.79:9090"
    ENDPOINT = "/modelstealing/reset"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "3FNWQO9kLVQmnLj4"

    response = requests.post(
        URL, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        print("Endpoint rested successfully")
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")
    
model_stealing_reset()