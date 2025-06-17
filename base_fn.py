import torch
import numpy as np

# 计算高斯分布的概率密度函数的对数值
def log_gaussian(x, mu, var):
    return -0.5 * (torch.log(torch.tensor(2.0 * np.pi)) + torch.log(var) + torch.pow(x - mu, 2) / var)

# 计算两个高斯分布之间的KL散度
def gaussian_kl(q_mu, q_var, p_mu, p_var):  # KL(Q|P)
    return - 0.5 * (torch.log(q_var / p_var) - q_var / p_var - torch.pow(q_mu - p_mu, 2) / p_var + 1)

# 估计潜在变量 z 属于每个高斯成分的概率
def vade_trick(mc_sample, mog_pi, mog_mu, mog_var):
    log_pz_c = torch.sum(log_gaussian(mc_sample.unsqueeze(1), mog_mu.unsqueeze(0), mog_var.unsqueeze(0)), dim=-1)
    log_pc = torch.log(mog_pi.unsqueeze(0))
    log_pc_z = log_pc + log_pz_c
    pc_z = torch.exp(log_pc_z) + 1e-10
    normalized_pc_z = pc_z / torch.sum(pc_z, dim=1, keepdim=True)
    return normalized_pc_z

# 计算整个模型的KL散度项，包括潜在变量 z 的KL散度以及类别变量 c 的KL散度
def kl_term(z_mu, z_var, qc_x, mog_weight, mog_mu, mog_var):
    z_kl_div = torch.sum(qc_x * torch.sum(gaussian_kl(z_mu.unsqueeze(1), z_var.unsqueeze(1), mog_mu.unsqueeze(0), mog_var.unsqueeze(0)), dim=-1),
                         dim=1)
    z_kl_div_mean = torch.mean(z_kl_div)

    c_kl_div = torch.sum(qc_x * torch.log(qc_x / mog_weight.unsqueeze(0)), dim=1)
    c_kl_div_mean = torch.mean(c_kl_div)
    return z_kl_div_mean, c_kl_div_mean

# 计算多视图数据中各个视图之间的相干性损失
def coherence_function(vs_mus, vs_vars, aggregated_mu, aggregated_var, mask=None):
    coherence_loss_list = []
    mask_stack = torch.cat(mask, dim=1)  # Batch size * V
    norm = torch.sum(mask_stack, dim=1)  # |Via|
    for v in range(len(vs_mus)):
        uniview_coherence_loss = torch.sum(gaussian_kl(aggregated_mu, aggregated_var, vs_mus[v], vs_vars[v]), dim=1)
        exist_loss = uniview_coherence_loss * mask[v].squeeze()
        coherence_loss_list.append(exist_loss)
    coherence_loss = torch.mean(sum(coherence_loss_list) / norm)
    return coherence_loss

import torch.nn as nn
import math
class ClusterLoss(nn.Module):
    """类簇级别的对比损失"""
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        # 确保 sim_i_j 和 sim_j_i 的长度都是 self.class_num
        assert sim_i_j.size(0) == self.class_num
        assert sim_j_i.size(0) == self.class_num
        
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + alpha * ne_loss
    
# 添加
import torch.nn.functional as F

def FL_Loss(clusters, h1, centers, label, h2=[], temperature=0.5):
    """
    Feature-level contrastive learning
    """
    loss = 0
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)

    indicator = torch.ones(h1.size(0), clusters).to(h1.device)
    for i in range(clusters):
        indicator[i, label[i]] = 0

    sim_positive = torch.exp(torch.mm(h1, h2.T) / temperature)
    sim_negative = torch.mul(
        torch.exp(torch.mm(h1, centers.T) / temperature), indicator)
    loss = - torch.log(torch.diagonal(sim_positive) /
                       (torch.diagonal(sim_positive) + torch.sum(sim_negative, dim=1)))
    return loss.mean()

# 通过信息瓶颈方法最小化不同视图之间的互信息
def compute_mutual_information(z1, z2, temperature=0.5):
    """
    Compute the mutual information between two views using InfoNCE loss.
    """
    sim_matrix = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    mutual_info = F.cross_entropy(sim_matrix, labels)
    return mutual_info
def FL_Loss_Mu(clusters, h1, centers, label, h2=[], temperature=0.2, lambda_mi=0.1):
    """
    Feature-level contrastive learning with mutual information minimization
    """
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)

    indicator = torch.ones(h1.size(0), clusters).to(h1.device)
    for i in range(clusters):
        indicator[i, label[i]] = 0

    sim_positive = torch.exp(torch.mm(h1, h2.T) / temperature)
    sim_negative = torch.mul(
        torch.exp(torch.mm(h1, centers.T) / temperature), indicator)

    if torch.diagonal(sim_positive).shape != torch.sum(sim_negative, dim=1).shape:
        raise ValueError("The shapes of sim_positive and sim_negative do not match.")

    contrastive_loss = - torch.log(torch.diagonal(sim_positive) /
                                   (torch.diagonal(sim_positive) + torch.sum(sim_negative, dim=1))).mean()

    # 计算互信息
    mutual_info = compute_mutual_information(h1, h2, temperature)

    # 添加互信息项到损失函数
    total_loss = contrastive_loss + lambda_mi * mutual_info

    return total_loss

class FeatureLoss(nn.Module):
    def __init__(self, temperature, device):
        super(FeatureLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N), device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, h_i, h_j, batch_size):
        N = 2 * batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature

        # # 调试信息
        # print(f"sim shape: {sim.shape}")

        # if sim.shape[0] != N:
        #     print("Warning: Incomplete batch detected. Skipping this batch.")
        #     return None

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # # 调试信息
        # print(f"sim_i_j shape: {sim_i_j.shape}, sim_j_i shape: {sim_j_i.shape}")

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        # # 调试信息
        # print(f"positive_samples shape: {positive_samples.shape}, negative_samples shape: {negative_samples.shape}")

        labels = torch.zeros(N, device=self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss