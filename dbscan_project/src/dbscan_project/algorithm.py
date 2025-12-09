import numpy as np
from scipy.spatial import distance


class CustomDBSCAN:
    """
    Custom implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.

    This algorithm identifies clusters based on the density of points in the data space.
    Points in low-density regions are marked as noise (-1).

    Attributes:
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        labels_ (np.ndarray): Cluster labels for each point in the dataset fit to the model. Noise is labeled as -1.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initializes the DBSCAN parameters.

        Args:
            eps (float): The radius of neighborhood (epsilon). Defaults to 0.5.
            min_samples (int): The minimum number of samples in the radius eps. Defaults to 5.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X: np.ndarray) -> 'CustomDBSCAN':
        """
        Perform DBSCAN clustering from vector array X.

        Args:
            X (np.ndarray): Input data array of shape (n_samples, n_features).

        Returns:
            self: Returns the instance of the object with calculated labels (self.labels_).
        """
        n_samples = X.shape[0]
        # Initialize labels: -1 represents Noise
        self.labels_ = np.full(n_samples, -1)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)

        # Calculate Euclidean distance matrix (Brute-force approach)
        # Note: Scikit-learn optimizes this using KD-Trees or Ball-Trees
        dist_matrix = distance.cdist(X, X, 'euclidean')

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True

            # Find neighbors within eps radius
            neighbors = np.where(dist_matrix[i] <= self.eps)[0]

            if len(neighbors) < self.min_samples:
                # Point is flagged as Noise (remains -1)
                continue
            else:
                # Start a new cluster
                self.labels_[i] = cluster_id
                self._expand_cluster(neighbors, cluster_id, visited, dist_matrix)
                cluster_id += 1
        return self

    def _expand_cluster(self, neighbors, cluster_id, visited, dist_matrix):
        """
        Recursively (or iteratively via queue) expands the cluster to include density-reachable points.

        Args:
            neighbors (np.ndarray): Array of indices of the initial neighbors.
            cluster_id (int): The ID of the current cluster being formed.
            visited (np.ndarray): Boolean array tracking visited points.
            dist_matrix (np.ndarray): Pre-calculated distance matrix.
        """
        # Use a list as a queue for processing neighbors
        queue = list(neighbors)

        i = 0
        while i < len(queue):
            point_idx = queue[i]

            if not visited[point_idx]:
                visited[point_idx] = True

                # Check neighbors of this neighbor
                new_neighbors = np.where(dist_matrix[point_idx] <= self.eps)[0]

                if len(new_neighbors) >= self.min_samples:
                    # Add new unique neighbors to the queue
                    existing_set = set(queue)
                    for n in new_neighbors:
                        if n not in existing_set:
                            queue.append(n)
                            existing_set.add(n)

            # If the point does not belong to any cluster yet, assign it to the current cluster
            if self.labels_[point_idx] == -1:
                self.labels_[point_idx] = cluster_id

            i += 1