import numpy as np
import random

def euclidean_distances(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def generate_initial_centroids(K, dimensionality):
    # Randomly generate K centroids with NumPy for better performance
    return np.random.uniform(0, 100, size=(K, dimensionality))

def assign_points_to_clusters(centroids, dataset):
    clusters = [[] for _ in range(len(centroids))]
    for point in dataset:
        # Compute distance to each centroid using NumPy for efficiency
        distances = np.sqrt(np.sum((centroids - point) ** 2, axis=1))
        # Assign to the closest centroid
        closest_centroid_index = np.argmin(distances)
        clusters[closest_centroid_index].append(point)
    return clusters

def update_centroids(clusters, dimensionality):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            # Reinitialize centroid if cluster is empty
            new_centroids.append(np.random.uniform(0, 100, dimensionality))
        else:
            # Calculate the mean for each dimension with NumPy
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid)
    return np.array(new_centroids)

def k_means(K, dataset, dimensionality, max_iterations=100):
    centroids = generate_initial_centroids(K, dimensionality)
    dataset = np.array(dataset)  # Convert dataset to NumPy array for efficient operations
    for _ in range(max_iterations):
        clusters = assign_points_to_clusters(centroids, dataset)
        new_centroids = update_centroids(clusters, dimensionality)
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

# Example usage:
if __name__ == "__main__":
    K = 10  
    dimensionality = 2  
    dataset_size = 10000  

    dataset = np.random.uniform(0, 100, (dataset_size, dimensionality))

    centroids, clusters = k_means(K, dataset, dimensionality)

