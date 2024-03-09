import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class KMeans:
    def normalize(self, x, dim):
        norm = torch.sqrt((x ** 2).sum(dim=dim, keepdim=True))
        return torch.zeros_like(x) if norm.sum() == 0 else x / norm

    # input : points   - N * C * H * W
    # return: centers  - K * C * H * W
    # return: indices  - N * C * H * W
    # return: clusters - (X * C * H * W) * K
    def __call__(self, points, num_centers, max_iters=100):
        points = self.normalize(points, dim=(1, 2, 3))
        centers = points[: num_centers]
        with tqdm(range(max_iters), dynamic_ncols=True) as kmeans_indices:
            for i in kmeans_indices:
                clusters = [[] for i in range(num_centers)]
                dists = []
                for c in centers:
                    dists.append(((c.unsqueeze(0) - points) ** 2).mean(dim=(1, 2, 3)).unsqueeze(-1))
                dists = torch.cat(dists, dim=-1)
                indices = dists.argmin(dim=-1)
                for j in range(num_centers):
                    clusters[j].append(points[indices == j])
                clusters = [torch.cat(c, dim=0) for c in clusters if len(c) > 0]
                new_centers = [c.mean(dim=0, keepdim=True) for c in clusters]
                if len(new_centers) < num_centers:
                    print(f'adding {num_centers - len(new_centers)} random centers at step {i}')
                for j in range(num_centers - len(new_centers)):
                    center = torch.randn_like(new_centers[0])
                    new_centers.append(center / center.norm())
                new_centers = torch.cat(new_centers, dim=0)
                center_shift = (centers - new_centers).abs().sum()
                kmeans_indices.set_postfix(ordered_dict={
                    'step': i,
                    'center shift': center_shift.item(),
                })
                kmeans_indices.set_description('failure mode clustering')
                if center_shift < 1e-10:
                    break
                centers = new_centers.clone()
        return centers, indices, clusters


class TSNE:
    def __init__(self, tsne_metric='euclidean'):
        assert tsne_metric in ['cosine', 'euclidean'], f'[ERROR] TSNE metric {tsne_metric} not supported'
        self.tsne = TSNE(n_components=2, init='pca', random_state=0, metric=tsne_metric)

    # input : points      - N * D
    # return: tsne_points - N * 2
    def __call__(self, points, fig_path=None, labels=None):
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        tsne_points = self.tsne.fit_transform(points)
        tsne_points = (tsne_points - tsne_points.min(axis=0)) / \
            (tsne_points.max(axis=0) - tsne_points.min(axis=0))
        assert fig_path is None or labels is not None, 'visualization requires valid labels'
        if fig_path is not None:
            fig = plt.figure()
            for i, p in enumerate(tsne_points):
                plt.text(p[0], p[1], str(labels[i]), color=plt.cm.Set1(labels[i] / 10), fontdict={'size': 10})
            plt.savefig(fig_path)
        return tsne_points

