import numpy as np

if __name__ == "__main__":
    data = np.load(
        "defensetransformation/data/ExampleDefenseTransformationEvaluate.npz"
    )
    print(data["labels"], data["representations"].shape)

    data = np.load("defensetransformation/data/ExampleDefenseTransformationSubmit.npz")
    print(data["representations"].shape)
