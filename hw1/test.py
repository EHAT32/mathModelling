import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def create_distance_matrix(data):
    n = data.shape[0]
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(data[i], data[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # Symmetric matrix

    return distance_matrix

def single_linkage_clustering(data):
    # Initialize clusters
    clusters = [[i] for i in range(len(data))]
    distance_matrix = create_distance_matrix(data)

    while len(clusters) > 1:
        # Find the closest pair of clusters
        min_distance = float('inf')
        closest_pair = (0, 0)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Calculate distance between cluster i and cluster j
                cluster_distance = min(distance_matrix[x][y] for x in clusters[i] for y in clusters[j])
                if cluster_distance < min_distance:
                    min_distance = cluster_distance
                    closest_pair = (i, j)

        # Merge the closest clusters
        cluster_i, cluster_j = closest_pair
        new_cluster = clusters[cluster_i] + clusters[cluster_j]
        clusters.append(new_cluster)

        # Update the distance matrix
        new_distances = []

        for k in range(len(clusters) - 1):  # Exclude the new cluster
            if k != cluster_i and k != cluster_j:
                # Calculate new distance using single linkage (minimum distance)
                new_distance = min(distance_matrix[x][y] for x in clusters[k] for y in new_cluster)
                new_distances.append(new_distance)

        # Create a new row with distances and self-distance as zero
        new_row = np.array(new_distances + [0])  # New row with distances to existing clusters + self-distance
        
        # Update the distance matrix with proper shape
        distance_matrix = np.vstack([distance_matrix, new_row])  # Add new row
        
        # Create a zero-filled column for distances from all existing clusters to the new cluster
        new_column = np.zeros(len(distance_matrix))  
        
        # Add the zero-filled column to the right of the updated distance matrix
        distance_matrix = np.column_stack([distance_matrix, new_column])

        # Remove old clusters from the list and update distances
        del clusters[max(cluster_i, cluster_j)]
        del clusters[min(cluster_i, cluster_j)]

    return clusters[0]  # Return the final merged cluster indices

# Example usage with a dataset of 10 samples and 5 numerical features
data = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.1, 2.1, 3.1, 4.1, 5.1],
    [2.0, 3.0, 4.0, 5.0, 6.0],
    [2.1, 3.1, 4.1, 5.1, 6.1],
    [10.0, 10.0, 10.0, 10.0, 10.0],
    [10.1, 10.1, 10.1, 10.1, 10.1],
    [11.0, 11.0, 11.0, 11.0, 11.0],
    [11.1, 11.1, 11.1, 11.1, 11.1],
    [5.0, 6.0, 7.0, 8.0, 9.0],
    [5.2, 6.2, 7.2, 8.2, 9.2],
])

final_cluster_indices = single_linkage_clustering(data)
print("Final Cluster Indices:", final_cluster_indices)