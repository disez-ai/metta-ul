import numpy as np
from heapq import heappush, heappop

def euclidean_distance_matrix(X):
    diffs = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diffs, axis=-1)

def cluster_distance(dist_matrix, c1, c2, linkage):
    d = dist_matrix[np.ix_(list(c1), list(c2))]
    if linkage == 'single':
        return np.min(d)
    elif linkage == 'complete':
        return np.max(d)
    elif linkage == 'average':
        return np.mean(d)
    else:
        raise ValueError("Invalid linkage type")

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.count = n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            return
        self.parent[rv] = ru
        self.count -= 1
    
    def __getitem__(self, index):
        root = self.find(index)
        return np.nonzero(np.equal(self.parent, root))[0]
    
    def __iter__(self):
        roots = np.unique(self.parent)
        for root in roots:
            yield root, self[root]

def hierarchical_clustering(X, k=2, linkage='single'):
    n = len(X)
    dist_matrix = euclidean_distance_matrix(X)

    # Initialize disjoint clusters and priority queue
    uf = UnionFind(n)
    heap = [(dist_matrix[i, j], i, j) for i in range(n) for j in range(i+1, n)]
    heap.sort()

    while uf.count > k:
        d, i, j = heappop(heap)
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue

        # Merge clusters
        uf.union(ri, rj)
        new_root = uf.find(ri)
        new_cluster = uf[new_root]

        # Update distances to all other clusters
        for root, cluster in uf:
            if root == new_root:
                continue
            d_new = cluster_distance(dist_matrix, new_cluster, cluster, linkage)
            heappush(heap, (d_new, new_root, root))

    # Build final labels
    label_map = {}
    labels = np.empty(n, dtype=int)
    current_label = 0
    for i in range(n):
        root = uf.find(i)
        if root not in label_map:
            label_map[root] = current_label
            current_label += 1
        labels[i] = label_map[root]
    return labels


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [1, 1],
        [1, 0],
        [5, 5],
        [10, 10],
        [5.5, 5.2]
    ])
    for linkage in ["single", "complete", "average"]:
        labels = hierarchical_clustering(X, k=4, linkage=linkage)
        print(f"{linkage.title()} linkage: {labels}")
