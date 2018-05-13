import operator
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances


class MSTClustering(object):
    """
    Minimum spanning tree clustering, also called single-linkage clustering
    Follow description in:
    (1) https://www.coursera.org/learn/algorithms-greedy/lecture/QWubN/application-to-clustering
    (2) https://en.wikipedia.org/wiki/Single-linkage_clustering
    """

    def __init__(self, num_cluster: int =2):
        self.num_cluster = (num_cluster-1)

    def fit(self, points):
        cluster_labels = np.zeros((points.shape[0], 1))

        data = pairwise_distances(points)
        G = nx.from_numpy_matrix(data)
        mst = nx.minimum_spanning_tree(G, weight='weight')

        # find edge to remove
        edge_list = [e for e in mst.edges.data()]
        edge_list = sorted(edge_list, key=lambda e: e[2]['weight'], reverse=True)

        # remove edge from mst
        for i in range(self.num_cluster):
            edge = edge_list[i]
            mst.remove_edge(edge[0], edge[1])

        # find clusters
        clusters = [c for c in nx.connected_components(mst)]
        cluster_idx = 1
        for cluster in clusters:
            cluster_labels[np.array(list(cluster))] = cluster_idx
            cluster_idx += 1

        self.visalize_cluster(points, cluster_labels)

    def visalize_cluster(self, points, cluster_labels):
        plt.scatter(points[:, 0], points[:, 1], c=cluster_labels[:, 0])
        plt.show()


if __name__ == '__main__':
    print("Test minimum spanning tree clustering")
    cluster_1 = np.hstack((np.random.normal(0, 1, size=(200, 1)), np.random.normal(0, 1, size=(200, 1))))
    cluster_2 = np.hstack((np.random.normal(0, 1, size=(200, 1)), np.random.normal(0, 1, size=(200, 1))))
    cluster_3 = np.hstack((np.random.normal(0, 1, size=(200, 1)), np.random.normal(0, 1, size=(200, 1))))

    cluster_1 /= np.sqrt(np.sum(cluster_1**2, axis=1)[:,  np.newaxis])
    cluster_2 /= 0.25 * np.sqrt(np.sum(cluster_2**2, axis=1)[:,  np.newaxis])
    cluster_3 /= 0.1 * np.sqrt(np.sum(cluster_3 ** 2, axis=1)[:, np.newaxis])

    points = np.vstack((cluster_1, cluster_2, cluster_3))

    mdl = MSTClustering(num_cluster=3)
    mdl.fit(points)


