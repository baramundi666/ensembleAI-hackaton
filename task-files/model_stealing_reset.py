def model_stealing_reset():
    SERVER_URL = "[paste server url here]"
    ENDPOINT = "/modelstealing/reset"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "[paste your team token here]"

    response = requests.post(
        URL, headers={"token": TEAM_TOKEN}
    )

    if response.status_code == 200:
        print("Endpoint rested successfully")
    else:
        raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")