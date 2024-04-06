import random

def euclidean_distances(point1, point2):
    return sum([(point1[i] - point2[i]) ** 2 for i in range(len(point1))]) ** 0.5

def generate_initial_centroids(K, dimensionality):
    # Randomly generate K centroids, assuming each dimension ranges from 0 to 100 for initialization
    return [[random.uniform(0, 100) for _ in range(dimensionality)] for _ in range(K)]

def assign_points_to_clusters(centroids, dataset):
    clusters = [[] for _ in centroids]
    for point in dataset:
        # Compute distance to each centroid
        distances = [euclidean_distances(point, centroid) for centroid in centroids]
        # Assign to the closest centroid
        closest_centroid_index = distances.index(min(distances))
        clusters[closest_centroid_index].append(point)
    return clusters

def update_centroids(clusters, dimensionality):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:  # Avoid division by zero
            new_centroids.append([random.uniform(0, 100) for _ in range(dimensionality)])  # Reinitialize if cluster is empty
        else:
            # Calculate the mean for each dimension
            new_centroid = [sum([point[dim] for point in cluster]) / len(cluster) for dim in range(dimensionality)]
            new_centroids.append(new_centroid)
    return new_centroids

def k_means(K, dataset, dimensionality, max_iterations=100):
    centroids = generate_initial_centroids(K, dimensionality)
    for _ in range(max_iterations):
        clusters = assign_points_to_clusters(centroids, dataset)
        new_centroids = update_centroids(clusters, dimensionality)
        if centroids == new_centroids:  # Check for convergence
            break
        centroids = new_centroids
    return centroids, clusters
