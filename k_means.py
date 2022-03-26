import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def k_means(points, k, max_iter=1000):
    centers = points[: k]
    for i in range(max_iter):
        clusters = [[] for i in range(k)]
        for point in points:
            distances = np.sum((point - centers) ** 2, axis=-1)
            min_dist_index = np.argmin(distances)
            clusters[min_dist_index].append(point)
        new_centers = np.zeros_like(centers)
        for j in range(len(centers)):
            new_centers[j] = np.mean(np.array(clusters[j]), axis=0)
        if np.sum((new_centers - centers) ** 2) < 1e-3:
            break
        if i % 10 == 0:
            print('iteration', i, 'error', np.sum((new_centers - centers) ** 2))
        centers = new_centers.copy()
    return new_centers


if __name__ == '__main__':
    num_samples = 200
    dimension = 2
    num_centers = 3
    data, target = make_blobs(n_samples=num_samples, n_features=dimension, centers=num_centers)
    centers = k_means(data, num_centers)
    plt.scatter(data[:, 0], data[:, 1], c=target)
    plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.savefig('result.png')

