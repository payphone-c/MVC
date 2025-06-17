import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from units import build_LPG, AttentionLayer, AttentionLayer_list, graph_normalize

# 从给定的均值（mu）和方差（var）中采样高斯分布的样本
class Gaussian_sampling(nn.Module):
    def forward(self, mu, var):
        # 计算标准差
        std = torch.sqrt(var)
        # 生成随机噪声
        epi = std.data.new(std.size()).normal_()
        # 生成高斯样本
        return epi * std + mu

# 根据多个视图的均值（mu）和方差（var）以及可选的掩码（mask）来聚合这些视图的高斯分布
class Gaussian_poe(nn.Module):
    def forward(self, mu, var, mask=None):
        mask_matrix = torch.stack(mask, dim=0)
        exist_mu = mu * mask_matrix
        T = 1. / var
        exist_T = T * mask_matrix
        aggregate_T = torch.sum(exist_T, dim=0)
        # 计算聚合方差
        aggregate_var = 1. / aggregate_T
        # 计算加权聚合均值
        aggregate_mu = torch.sum(exist_mu * exist_T, dim=0) / aggregate_T
        return aggregate_mu, aggregate_var

# 将来自特定视图的输入数据编码成潜在变量的均值和方差
class view_specific_encoder(nn.Module):
    def __init__(self, view_dim, latent_dim, class_num):
        super(view_specific_encoder, self).__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.k = class_num
        self.encoder = nn.Sequential(nn.Linear(self.x_dim, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 2000),
                                     nn.ReLU()
                                     )
        self.z_mu = nn.Linear(2000, self.z_dim)
        self.z_var = nn.Sequential(nn.Linear(2000, self.z_dim), nn.Softplus())
        # 添加高维特征
        self.high_feature_contrastive_module = nn.Sequential(
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
        )
        # _________________test____________________
        # # 简单修改，获取簇级潜在表示
        # self.cluster = nn.Sequential(
        #     nn.Linear(self.z_dim, self.k),
        #     # nn.BatchNorm1d(self.k), 
        #     nn.Softmax(dim=1)
        # )
        # 改进后的 cluster 模块
        self.cluster = nn.Sequential(
            nn.Linear(self.z_dim, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 添加 Dropout
            nn.Linear(50, self.k),
            nn.Softmax(dim=1)
        )
        self.cluster1 = nn.Sequential(
            nn.Linear(self.z_dim, 50),
            # nn.BatchNorm1d(50),  # 若batch_size为1,出去此层
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 添加 Dropout
            nn.Linear(50, self.k),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        # 将输入数据 x 通过编码器网络，得到隐藏特征
        hidden_feature = self.encoder(x)
        vs_mu = self.z_mu(hidden_feature)
        vs_var = self.z_var(hidden_feature)
        # 
        high_feature = normalize(self.high_feature_contrastive_module(hidden_feature), dim=1)
        # 修改 输入为latent_represent
        if(vs_mu.size(0)>1) :
            y = self.cluster(vs_mu)
        else:
            y = self.cluster1(vs_mu)
        # 返回潜在变量的均值和方差
        return vs_mu, vs_var, y, high_feature, hidden_feature
# 将潜在空间的表示映射回原始数据空间 ,重构输入数据
class view_specific_decoder(nn.Module):
    def __init__(self, view_dim, latent_dim):
        super(view_specific_decoder, self).__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.decoder = nn.Sequential(nn.Linear(self.z_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, self.x_dim),
                                     )

    def forward(self, z):
        xr = self.decoder(z)
        return xr
# Mixture of Experts (MoE)方法
class Gaussian_moe(nn.Module):
    def forward(self, mus, vars, mask=None):
        # 计算加权系数 w_i = 1 / var_i
        precisions = [1. / var for var in vars]
        
        # 将 mask 堆叠
        mask_matrix = torch.stack(mask, dim=0)
        
        # 应用 mask，保留有效视角
        exist_mus = [mu * mask_matrix[i] for i, mu in enumerate(mus)]
        exist_precisions = [precision * mask_matrix[i] for i, precision in enumerate(precisions)]
        
        # 聚合方差: aggregate_var = 1 / sum(precisions)
        aggregate_precision = sum(exist_precisions)  # sum(w_i)
        aggregate_var = 1. / aggregate_precision
        
        # 聚合均值: aggregate_mu = sum(w_i * mu_i) / sum(w_i)
        weighted_mus = [exist_mu * exist_precision for exist_mu, exist_precision in zip(exist_mus, exist_precisions)]
        aggregate_mu = sum(weighted_mus) / aggregate_precision
        
        return aggregate_mu, aggregate_var
class DVIMC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.x_dim_list = args.multiview_dims
        self.k = args.class_num
        self.z_dim = args.z_dim
        self.num_views = args.num_views

        self.prior_weight = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)
        self.prior_mu = nn.Parameter(torch.full((self.k, self.z_dim), 0.0), requires_grad=True)
        self.prior_var = nn.Parameter(torch.full((self.k, self.z_dim), 1.0), requires_grad=True)

        self.encoders = nn.ModuleDict({f'view_{v}': view_specific_encoder(self.x_dim_list[v], self.z_dim, self.k) for v in range(self.num_views)})
        self.decoders = nn.ModuleDict({f'view_{v}': view_specific_decoder(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)})

        self.aggregated_fn = Gaussian_poe()
        # moe方法
        self.aggregated_moe = Gaussian_moe()
        self.sampling_fn = Gaussian_sampling()
        self.fusion_list = AttentionLayer_list(2000,self.num_views)
        self.device = args.device
        self.hidden_size=args.hidden_size
    def attention_fusion_list(self, h_list):
        h = self.fusion_list(h_list)
        return h
    # 对来自多个视图的数据进行编码，生成每个视图的潜在变量的表示,及簇级表示
    def mv_encode(self, x_list):
        latent_representation_list = []
        y_list = []
        h_list = []
        hidden_list = []
        for v in range(self.num_views):
            latent_representation, _, y, h, hidden = self.encoders[f'view_{v}'](x_list[v])
            latent_representation_list.append(latent_representation)
            y_list.append(y)
            h_list.append(h)
            hidden_list.append(hidden)
        return latent_representation_list, y_list, h_list, hidden_list

    # 对来自特定视图的输入数据 x 进行编码和解码，生成该视图的潜在变量表示和重构数据
    def sv_encode(self, x, view_idx):
        latent_representation, _, _, _, _ = self.encoders[f'view_{view_idx}'](x)
        xr = self.decoders[f'view_{view_idx}'](latent_representation)
        return latent_representation, xr

    def inference_z(self, x_list, mask):
        vs_mus, vs_vars = [], []
        for v in range(self.num_views):
            vs_mu, vs_var, _, _, _ = self.encoders[f'view_{v}'](x_list[v])
            vs_mus.append(vs_mu)
            vs_vars.append(vs_var)
        mu = torch.stack(vs_mus)
        var = torch.stack(vs_vars)
        aggregated_mu, aggregated_var = self.aggregated_fn(mu, var, mask)
        # aggregated_mu, aggregated_var = self.aggregated_moe(mu, var, mask)
        return vs_mus, vs_vars, aggregated_mu, aggregated_var
        # 返回每个视图的潜在变量均值和方差，以及聚合后的全局潜在变量均值和方差。

    # 根据潜在变量 z 生成每个视图的重构数据
    def generation_x(self, z):
        xr_list = [vs_decoder(z) for vs_decoder in self.decoders.values()]
        return xr_list

    def forward(self, x_list, mask=None):
        vs_mus, vs_vars, aggregated_mu, aggregated_var = self.inference_z(x_list, mask)
        z_sample = self.sampling_fn(aggregated_mu, aggregated_var)
        xr_list = self.generation_x(z_sample)
        vade_z_sample = self.sampling_fn(aggregated_mu, aggregated_var)
        return z_sample, vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample