import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def compute_sse(data, indices, center):
    pts = data[indices]
    return np.sum((pts - center) ** 2)


def bisect_cluster(data, cluster, n_init=10):
    # cluster is a tuple: (indices, center, sse, children)
    indices = cluster[0]
    pts = data[indices]
    kmeans = KMeans(n_clusters=2, n_init=n_init, random_state=42)
    labels = kmeans.fit_predict(pts)
    centers = kmeans.cluster_centers_

    sub0 = indices[np.where(labels == 0)[0]]
    sub1 = indices[np.where(labels == 1)[0]]

    # Create new clusters as tuples: (indices, center, sse, children)
    cluster0 = (sub0, centers[0], compute_sse(data, sub0, centers[0]), None)
    cluster1 = (sub1, centers[1], compute_sse(data, sub1, centers[1]), None)
    return cluster0, cluster1


def remove_cluster(clusters, target):
    if clusters == []:
        return []
    head = clusters[0]
    tail = remove_cluster(clusters[1:], target)
    return tail if head is target else [head] + tail


def find_max_cluster(clusters):
    # Returns the cluster (tuple) with the maximum SSE (third element)
    if len(clusters) == 1:
        return clusters[0]
    head = clusters[0]
    rest_max = find_max_cluster(clusters[1:])
    return head if head[2] >= rest_max[2] else rest_max


def recursive_bisecting_hierarchy(data, clusters, max_clusters, n_init=10, hierarchy=None):
    """
    Recursively bisects the cluster with maximum SSE until reaching max_clusters.
    Returns a list where each element is a list of clusters at that level.
    The last element in the list is the final set of clusters.
    """
    if hierarchy is None:
        hierarchy = [clusters]
    if len(clusters) >= max_clusters:
        return hierarchy
    # Find the cluster with the maximum SSE.
    max_cluster = find_max_cluster(clusters)
    # Remove that cluster from the current list.
    clusters = remove_cluster(clusters, max_cluster)
    # Bisect the selected cluster.
    child0, child1 = bisect_cluster(data, max_cluster, n_init)
    # Optionally, you could record the children in the parent's structure by creating a new tuple,
    # but since we're replacing the parent, we simply add the children.
    new_clusters = clusters + [child0, child1]
    hierarchy.append(new_clusters)
    # Recurse until reaching the desired number of clusters.
    return recursive_bisecting_hierarchy(data, new_clusters, max_clusters, n_init, hierarchy)


def assign_labels_recursive(clusters, labels, current=0):
    if clusters == []:
        return labels
    # cluster[0] is the indices array.
    indices = clusters[0][0]
    labels[indices] = current
    return assign_labels_recursive(clusters[1:], labels, current + 1)


def get_final_labels(clusters, n_samples):
    labels = np.empty(n_samples, dtype=int)
    return assign_labels_recursive(clusters, labels)


def plot_centers(clusters, idx=0):
    if idx >= len(clusters):
        return
    # cluster[1] is the center.
    center = clusters[idx][1]
    plt.scatter(center[0], center[1], c='red', s=100, marker='x')
    plot_centers(clusters, idx + 1)


def initial_cluster(data):
    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    center = np.mean(data, axis=0)
    sse = compute_sse(data, indices, center)
    # Return a list with a single cluster tuple.
    return [(indices, center, sse, None)]


def fit_bisecting_kmeans(X, n_clusters, n_init=10):
    """
    Functional 'fit' for bisecting k-means.
    Returns final clusters and labels.
    """
    init_clusters = initial_cluster(X)
    hierarchy = recursive_bisecting_hierarchy(X, init_clusters, n_clusters, n_init)
    final_clusters = hierarchy[-1]
    labels = get_final_labels(final_clusters, X.shape[0])
    return final_clusters, labels


def assign_point_to_cluster(point, clusters, best_idx=0, best_dist=None, idx=0):
    if idx >= len(clusters):
        return best_idx
    center = clusters[idx][1]
    dist = np.linalg.norm(point - center)
    if best_dist is None or dist < best_dist:
        return assign_point_to_cluster(point, clusters, idx, dist, idx + 1)
    return assign_point_to_cluster(point, clusters, best_idx, best_dist, idx + 1)


def assign_all_points(X, clusters, idx=0, labels=None):
    if idx == 0:
        labels = np.empty(X.shape[0], dtype=int)
    if idx >= X.shape[0]:
        return labels
    point = X[idx]
    cluster_idx = assign_point_to_cluster(point, clusters)
    labels[idx] = cluster_idx
    return assign_all_points(X, clusters, idx + 1, labels)


def predict_bisecting_kmeans(clusters, X_new):
    """
    Functional 'predict' for bisecting k-means.
    Assigns new points to the nearest cluster center.
    """
    return assign_all_points(X_new, clusters)


def main():
    # Create synthetic data.
    X, _ = make_blobs(n_samples=550, centers=4, cluster_std=0.60, random_state=42)
    X_test = X[-50:, :]
    X = X[:-50, :]

    # Fit
    final_clusters, labels = fit_bisecting_kmeans(X, n_clusters=4)

    # Score
    score = silhouette_score(X, labels)
    print("Silhouette Score: {:.3f}".format(score))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', alpha=0.7)
    plot_centers(final_clusters)
    plt.title("Bisecting KMeans Hierarchy (Final Clusters in Last Level)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Predict on new data
    new_labels = predict_bisecting_kmeans(final_clusters, X_test)
    print("Predicted labels for new samples:", new_labels)

    # Plot new points
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', alpha=0.5)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=new_labels, s=100, edgecolor='k', cmap='viridis', marker='D')
    plot_centers(final_clusters)
    plt.title("Predicted Labels for New Data Points")
    plt.show()

    # Print hierarchy info
    print("Final Clusters:", len(final_clusters))


if __name__ == "__main__":
    main()
