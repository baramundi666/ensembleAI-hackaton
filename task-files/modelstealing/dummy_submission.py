import os
import torch
import torch.nn as nn

if __name__ == "__main__":
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 512),
    )
    os.makedirs("modelstealing/models", exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        "modelstealing/models/example_submission.onnx",
        export_params=True,
        input_names=["x"],
    )
