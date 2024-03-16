import os
import numpy as np

if __name__ == "__main__":
    np.savez(
        "sybilattack/data/example_submission.npz",
        ids=np.random.permutation(20000),
        representations=np.random.randn(20000, 192),
    )
