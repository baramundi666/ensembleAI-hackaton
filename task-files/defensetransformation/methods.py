from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import shuffle
import numpy as np

def basic(data):
    desired_size = np.random.randint(32, 192)

    # Perform dimensionality reduction using Randomized PCA
    pca = PCA(n_components=desired_size, random_state=43)
    transformed_data = pca.fit_transform(data)

    # Quantize the transformed representations
    quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=41)
    transformed_data = quantile_transformer.fit_transform(transformed_data)

    # Additive Gaussian Noise
    noise = np.random.normal(0, 0.05, transformed_data.shape)
    transformed_data += noise

    print(transformed_data)

    # Feature Shuffling
    transformed_data = shuffle(transformed_data, random_state=42)


    return transformed_data

def second(data):
    target_dim = 512
    X=data
    # Step 1: Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(X.shape[1], target_dim))
    X_pca = pca.fit_transform(X)
    
    # Step 2: Random projection to further reduce dimensionality
    if X_pca.shape[1] > target_dim:
        random_projection = GaussianRandomProjection(n_components=target_dim)
        X_protected = random_projection.fit_transform(X_pca)
    else:
        X_protected = X_pca

    return X_protected