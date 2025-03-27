import numpy as np

def euclidean_distance_matrix(points):
    """Compute the pairwise Euclidean distance matrix using vectorized operations."""
    points = np.array(points)
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diffs**2, axis=-1))

def linkage_distance(dist_matrix, cluster1, cluster2, linkage):
    """Compute linkage distance between two clusters based on the specified method."""
    distances = dist_matrix[np.ix_(cluster1, cluster2)]
    if linkage == 'single':
        return np.min(distances)
    elif linkage == 'complete':
        return np.max(distances)
    elif linkage == 'average':
        return np.mean(distances)
    else:
        raise ValueError("Invalid linkage type. Choose 'single', 'complete', or 'average'.")

def merge_clusters(clusters, dist_matrix, linkage):
    """Find and merge the two closest clusters based on the linkage criterion."""
    n = len(clusters)
    min_dist = float('inf')
    merge_pair = None

    # Find the closest pair of clusters
    for i in range(n):
        for j in range(i + 1, n):
            dist = linkage_distance(dist_matrix, clusters[i], clusters[j], linkage)
            if dist < min_dist:
                min_dist = dist
                merge_pair = (i, j)

    # Merge the closest clusters
    i_min, j_min = merge_pair
    new_cluster = clusters[i_min] + clusters[j_min]

    # Create a new list of clusters excluding merged ones
    new_clusters = [clusters[k] for k in range(n) if k not in (i_min, j_min)]
    new_clusters.append(new_cluster)

    return new_clusters, ((clusters[i_min], clusters[j_min]), min_dist)

def hierarchical_clustering(points, linkage='single'):
    """
    Perform bottom-up hierarchical clustering using recursion and NumPy.
    
    Parameters:
        points (list): A list of points (each point is a NumPy array or list of coordinates).
        linkage (str): Linkage method: 'single', 'complete', or 'average'.
    
    Returns:
        A tuple (final_cluster, merge_history) where:
          - final_cluster is the single cluster containing all points.
          - merge_history is a list of tuples recording the merged clusters and their distances.
    """
    # Compute pairwise Euclidean distance matrix
    dist_matrix = euclidean_distance_matrix(points)
    
    # Start with each point as its own cluster (store indices)
    clusters = [[i] for i in range(len(points))]
    merge_history = []

    def recursive_merge(clusters, merge_history):
        if len(clusters) == 1:
            return clusters[0], merge_history

        new_clusters, merge_info = merge_clusters(clusters, dist_matrix, linkage)
        merge_history.append(merge_info)

        return recursive_merge(new_clusters, merge_history)

    final_cluster, history = recursive_merge(clusters, merge_history)
    
    # Convert indices back to original points
    final_cluster = [points[i] for i in final_cluster]

    return final_cluster, history

# Example usage:
if __name__ == '__main__':
    # Define a simple dataset (each point is a NumPy array or tuple of coordinates).
    data_points = np.array([
        [1, 2],
        [2, 3],
        [5, 8],
        [8, 8],
        [1, 0]
    ])
    
    # Perform clustering using different linkage criteria.
    for method in ['single', 'complete', 'average']:
        final_cluster, history = hierarchical_clustering(data_points, linkage=method)
        print(f"\nLinkage: {method}")
        print("Merge History (merged clusters and distance at merge):")
        for merge, dist in history:
            print(f"Merge: {merge}, Distance: {dist:.3f}")
        print("Final Cluster:", final_cluster)
