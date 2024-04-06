import numpy as np
from memory_profiler import profile
import kmeans_cython 

class Cy_KMeans:
    def __init__(self, k):
        self.k = k
        self.dimensionality = None
        self.centroids = None

    def fit(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data is not array")
        self.dimensionality = data.shape[1]
        self.centroids, _ = kmeans_cython.k_means(self.k, data, self.dimensionality, max_iterations=100)

cy_data = np.random.randint(0, 101, size=(30000, 2))

cy_kmeans = Cy_KMeans(k=5)

@profile
def run_kmeans():
    cy_kmeans.fit(cy_data)

if __name__ == "__main__":
    run_kmeans()
