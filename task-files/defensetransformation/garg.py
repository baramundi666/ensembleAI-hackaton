from methods import basic
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    url = "http://34.71.138.79:9090" + endpoint
    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": "3FNWQO9kLVQmnLj4"})
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
        else:
            raise Exception(
                f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
            )

def evaluate_test(function):
    data = np.load("data/DefenseTransformationEvaluate.npz")
    print(data['representations'])
    

    X_clean = data['representations']
    X_representations=function(X_clean)
    y = data['labels']

    # Splitting the data into train and test sets
    X_clean_train, X_clean_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)
    X_rep_train, X_rep_test, _, _ = train_test_split(X_representations, y, test_size=0.2, random_state=42)

    # Training classifier on clean representations
    classifier_clean = LogisticRegression(max_iter=1000)  # You can use any classifier of your choice
    classifier_clean.fit(X_clean_train, y_train)

    # Predicting and calculating accuracy for clean representations
    y_pred_clean = classifier_clean.predict(X_clean_test)
    accuracy_clean = accuracy_score(y_test, y_pred_clean)
    print("Accuracy with clean representations:", accuracy_clean)

    # Training classifier on representations
    classifier_rep = LogisticRegression(max_iter=1000)  # You can use any classifier of your choice
    classifier_rep.fit(X_rep_train, y_train)

    # Predicting and calculating accuracy for representations
    y_pred_rep = classifier_rep.predict(X_rep_test)
    accuracy_rep = accuracy_score(y_test, y_pred_rep)
    print("Accuracy with representations:", accuracy_rep)


    
def submit_solution(function):
    submition_data = np.load(
        "data/DefenseTransformationSubmit.npz")['representations']
    T = function(submition_data)
    

    np.savez_compressed('file.npz', representations=T)
    defense_submit("file.npz")

if __name__ == "__main__":
    #submit_solution(basic)
    evaluate_test(basic)



    

  