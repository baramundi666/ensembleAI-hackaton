from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
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