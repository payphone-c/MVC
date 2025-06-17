import torch
import torch.nn as nn
import torch.nn.functional as F

# 构建局部概率图 (LPG)。该函数是解决聚类问题的一种方法，称为 CAN（Clustering-with-Adaptive-Neighbors）
def build_LPG(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).to(X.device)
        weights += torch.eye(size).to(X.device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to(X.device)
    weights = weights.to(X.device)
    return weights
    # 返回对称的权重矩阵和原始权重矩阵

def graph_normalize(A):
    """
    Normalize the adj matrix
    """
    degree = torch.sum(A, dim=1).pow(-0.5)
    return (A * degree).t() * degree

def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


# 实现一个注意力机制层，用于在两个向量 h1 和 h2 之间分配注意力权重，并根据这些权重对向量进行加权组合
class AttentionLayer(nn.Module):
    def __init__(self, latent_dim):
        super(AttentionLayer, self).__init__()
        self._latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(self._latent_dim * 2, self._latent_dim * 2),
            nn.BatchNorm1d(self._latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self._latent_dim * 2, self._latent_dim * 2),
            nn.BatchNorm1d(self._latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.output_layer = nn.Linear(self._latent_dim * 2, 2, bias=True)

    def forward(self, h1, h2, tau=10.0):
        h = torch.cat((h1, h2), dim=1)
        act = self.output_layer(self.mlp(h))
        act = F.sigmoid(act) / tau
        e = F.softmax(act, dim=1)
        # weights = torch.mean(e, dim=0)
        # h = weights[0] * h1 + weights[1] * h2
        h = e[:, 0].unsqueeze(1) * h1 + e[:, 1].unsqueeze(1) * h2
        return h
# 
class AttentionLayer_list(nn.Module):
    def __init__(self, latent_dim, num_views):
        super(AttentionLayer_list, self).__init__()
        self._latent_dim = latent_dim
        self._num_views = num_views
        self.mlp = nn.Sequential(
            nn.Linear(self._latent_dim * num_views, self._latent_dim * 2),
            nn.BatchNorm1d(self._latent_dim * 2),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(self._latent_dim * 2, self._latent_dim * 2),
            # nn.BatchNorm1d(self._latent_dim * 2),
            # nn.ReLU(),
            # # nn.Dropout(0.5),
        )
        self.output_layer = nn.Linear(self._latent_dim * 2, num_views, bias=True)

    def forward(self, h_list, tau=10.0):
        # 将所有潜在变量拼接在一起
        h = torch.cat(h_list, dim=1)
        
        # 通过 MLP 获取注意力权重
        act = self.output_layer(self.mlp(h))
        act = F.sigmoid(act) / tau
        e = F.softmax(act, dim=1)
        
        # 根据注意力权重融合潜在变量
        fused_h = torch.sum(e.unsqueeze(2) * torch.stack(h_list, dim=1), dim=1)
        
        return fused_h