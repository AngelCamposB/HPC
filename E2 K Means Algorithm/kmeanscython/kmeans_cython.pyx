# kmeans_cy.pyx
import numpy as np
cimport numpy as cnp

def euclidean_distances(cnp.ndarray[cnp.float64_t, ndim=1] point1, cnp.ndarray[cnp.float64_t, ndim=1] point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def generate_initial_centroids(int K, int dimensionality):
    return np.random.uniform(0, 100, size=(K, dimensionality))

def assign_points_to_clusters(cnp.ndarray centroids, cnp.ndarray dataset):
    cdef list clusters = [[] for _ in range(len(centroids))]
    cdef int closest_centroid_index
    for point in dataset:
        distances = np.sqrt(np.sum((centroids - point) ** 2, axis=1))
        closest_centroid_index = np.argmin(distances)
        clusters[closest_centroid_index].append(point.tolist())
    return clusters

def update_centroids(clusters, int dimensionality):
    cdef list new_centroids = []
    for cluster in clusters:
        if not cluster:
            new_centroids.append(np.random.uniform(0, 100, dimensionality).tolist())
        else:
            new_centroids.append(np.mean(cluster, axis=0).tolist())
    return np.array(new_centroids)

def k_means(int K, cnp.ndarray dataset, int dimensionality, int max_iterations=100):
    cdef cnp.ndarray centroids = generate_initial_centroids(K, dimensionality)
    for _ in range(max_iterations):
        clusters = assign_points_to_clusters(centroids, dataset)
        centroids = update_centroids(clusters, dimensionality)
    return centroids, clusters
