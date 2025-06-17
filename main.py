# 文件头部强制插入
import os
os.environ.update({
    'LOKY_MAX_CPU_COUNT': '4',  # 物理核心数
    'KMP_AFFINITY': 'disabled',
    'MKL_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1'
})

# 添加警告过滤
import warnings
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
    module="joblib"
)

# 修改KMeans调用
from sklearn.cluster import KMeans

class SafeKMeans:
    """安全封装KMeans"""
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def fit(self, data):
        """强制单线程执行"""
        return KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            n_jobs=1  # 显式禁用并行
        ).fit(data)
import copy
from scipy import cluster
import torch
from torch import optim
from torch.utils.data import DataLoader
from datasets import build_dataset
import torch.nn as nn
from base_model import DVIMC
from evaluate import evaluate
from base_fn import kl_term, vade_trick, coherence_function, ClusterLoss, FeatureLoss,FL_Loss, FL_Loss_Mu
import numpy as np
import random
import argparse
from sklearn.cluster import KMeans

from units import build_LPG, graph_normalize
import torch.nn.functional as F
import setproctitle

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# 定义线程名
setproctitle.setproctitle('chenc')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"  # 避免KMeans内存泄漏

import mkl
mkl.set_num_threads(1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def initialization(model, sv_loaders, cmv_data, args):
    print('Initializing......')
    criterion = nn.MSELoss()
    for v in range(args.num_views):
        optimizer = optim.Adam([{"params": model.encoders[f'view_{v}'].parameters(), 'lr': 0.001},
                                {"params": model.decoders[f'view_{v}'].parameters(), 'lr': 0.001},
                                ])
        for e in range(1, args.initial_epochs + 1):
            for batch_idx, xv in enumerate(sv_loaders[v]):
                optimizer.zero_grad()
                batch_size = xv.shape[0]
                xv = xv.reshape(batch_size, -1).to(args.device)
                _, xvr = model.sv_encode(xv, v)
                view_rec_loss = criterion(xvr, xv)
                view_rec_loss.backward()
                optimizer.step()
    with torch.no_grad():
        initial_data = [torch.tensor(csv_data, dtype=torch.float32).to(args.device) for csv_data in cmv_data]
        latent_representation_list, _, _, _ = model.mv_encode(initial_data)
        assert len(latent_representation_list) == args.num_views
        fused_latent_representations = sum(latent_representation_list) / len(latent_representation_list)
        fused_latent_representations = fused_latent_representations.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=args.class_num, n_init=10)
        kmeans.fit(fused_latent_representations)
        model.prior_mu.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(args.device)


def train(model, optimizer, scheduler, imv_loader, args, t):
    print('Training......')
    eval_data = copy.deepcopy(imv_loader.dataset.data_list)
    eval_mask = copy.deepcopy(imv_loader.dataset.mask_list)
    for v in range(args.num_views):
        eval_data[v] = torch.tensor(eval_data[v], dtype=torch.float32).to(args.device)
        eval_mask[v] = torch.tensor(eval_mask[v], dtype=torch.float32).to(args.device)
    eval_labels = imv_loader.dataset.labels

    if args.likelihood == 'Bernoulli':
        likelihood_fn = nn.BCEWithLogitsLoss(reduction='none')
    else:
        likelihood_fn = nn.MSELoss(reduction='none')
    # 添加簇对比
    criterion_cluster = ClusterLoss(args.class_num, 0.5, args.device).to(args.device)
    # criterion_instance = FeatureLoss(0.5, args.device).to(args.device)
    # 提取最大ACC
    eval_list = []
    for epoch in range(1, args.epochs + 1):
        epoch_loss = []
        for batch_idx, (batch_data, batch_mask) in enumerate(imv_loader):
            optimizer.zero_grad()
            batch_data = [sv_d.to(args.device) for sv_d in batch_data]
            batch_mask = [sv_m.to(args.device) for sv_m in batch_mask]
            z_sample, vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample = model(batch_data, batch_mask)
           
            # # 获取多个潜在表示列表
            _, y_list, _, hidden_list = model.mv_encode(batch_data)

            # 簇对比
            c_list = []
            for v in range(args.num_views):
                for w in range(v+1,args.num_views):
                    c_loss = criterion_cluster(y_list[v],y_list[w])
                    c_list.append(c_loss)
            cluster_loss = sum(c_list) 



            # 特征级别对比
            mean_h = sum(hidden_list) / len(hidden_list)
            # 注意力机制
            fused_h = model.attention_fusion_list(hidden_list)
            km = KMeans(n_clusters=args.class_num, n_init=10,
                            init='k-means++').fit(fused_h.data.cpu().numpy())
            label = torch.LongTensor(km.labels_).to(args.device)
            centers = torch.FloatTensor(km.cluster_centers_).to(args.device)

            # test
            feature_loss = FL_Loss(args.class_num,mean_h,centers,label,fused_h)

            # ------------------------------
            # # 特征级别对比
            # ins_loss = []
            # # fused_h = sum(hidden_list) / len(hidden_list)
            # # 注意力机制
            # fused_h = model.attention_fusion_list(hidden_list)
            # km = KMeans(n_clusters=args.class_num, n_init=10,
            #                 init='k-means++').fit(fused_h.data.cpu().numpy())
            # label = torch.LongTensor(km.labels_).to(args.device)
            # centers = torch.FloatTensor(km.cluster_centers_).to(args.device)

            # # test1-best
            # for v in range(args.num_views-1):
            #     in_loss = FL_Loss_Mu(args.class_num,hidden_list[v],centers,label,fused_h)
            #     ins_loss.append(in_loss)
            # feature_loss = sum(ins_loss) / len(ins_loss)
            # ------------------------------
            # # 实例级对比
            # fea_loss = []
            # for v in range(args.num_views-1):
            #     for w in range(v+1, args.num_views):
            #         actual_batch_size = high_list[v].size(0)
            #         in_loss = criterion_instance(high_list[v], high_list[w], actual_batch_size)
            #         fea_loss.append(in_loss)
            # feature_loss = sum(fea_loss)
            # ####
            qc_x = vade_trick(vade_z_sample, model.prior_weight, model.prior_mu, model.prior_var)
            z_loss, c_loss = kl_term(aggregated_mu, aggregated_var, qc_x, model.prior_weight, model.prior_mu, model.prior_var)
            kl_loss = z_loss + c_loss

            rec_term = []
            for v in range(args.num_views):
                sv_rec = torch.sum(likelihood_fn(xr_list[v], batch_data[v]), dim=1)  # ( Batch size * Dv )
                exist_rec = sv_rec * batch_mask[v].squeeze()
                view_rec_loss = torch.mean(exist_rec)
                rec_term.append(view_rec_loss)
            rec_loss = sum(rec_term)

            coherence_loss = coherence_function(vs_mus, vs_vars, aggregated_mu, aggregated_var, batch_mask)
            # 在此处添加对比损失
            batch_loss = rec_loss + kl_loss + args.alpha * coherence_loss + args.lamb3 * feature_loss + args.lamb0 * cluster_loss #
                                            
                                            # + args.lamb1 * feature_loss  + args.lamb0 * cluster_loss 

            epoch_loss.append(batch_loss.item())
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            model.prior_weight.data = model.prior_weight.data / model.prior_weight.data.sum()
        scheduler.step()
        overall_loss = sum(epoch_loss) / len(epoch_loss)

        if epoch % args.interval == 0 or epoch == args.epochs:
            with torch.no_grad():
                _, h1_list, _, aggregated_mu, _, _, _ = model(eval_data, eval_mask)

                mog_weight = model.prior_weight.data.detach().cpu()
                mog_mu = model.prior_mu.data.detach().cpu()
                mog_var = model.prior_var.data.detach().cpu()
                aggregated_mu = aggregated_mu.detach().cpu()

                c_assignment = vade_trick(aggregated_mu, mog_weight, mog_mu, mog_var)
                predict = torch.argmax(c_assignment, dim=1).numpy()
                acc, nmi, ari, pur = evaluate(eval_labels, predict)
                print(f'Epoch {epoch:>3}/{args.epochs}  Loss:{overall_loss:.2f}  ACC:{acc * 100:.2f}  '
                      f'NMI:{nmi * 100:.2f}  ARI:{ari * 100:.2f}  PUR:{pur * 100:.2f}')
                eval_list.append((acc, nmi, ari, pur))
                # # 可视化
                # # 使用T-SNE将数据降维到2D
                # # h_list_1, y_list, _, _ = model.mv_encode(eval_data)
                # tsne = TSNE(n_components=2, random_state=21) # 42
                # # 假设 z_list 包含所有批次的数据
                # # all_data = torch.cat(h1_list).detach().cpu().numpy()
                # # data_2d = tsne.fit_transform(all_data)
                # fused_y = sum(h1_list) / len(h1_list)
                # data_2d = tsne.fit_transform(fused_y.detach().cpu().numpy())
                # plt.figure(figsize=(20, 10))
                # plt.subplot(121)
                # plt.scatter(data_2d[:, 0], data_2d[:, 1], c=predict, label="t-SNE")
                # plt.legend()
                # plt.savefig(f"log/{args.dataset_name}_{t}_{epoch/args.interval}.png", dpi=500)
                # plt.show()
    acc, nmi, ari, pur = max(eval_list, key=lambda x: x[0])
    # # 使用T-SNE将数据降维到2D
    # h_list_1, y_list, _, _ = model.mv_encode(eval_data)
    # tsne = TSNE(n_components=2, random_state=42)
    # # 假设 z_list 包含所有批次的数据
    # # all_data = torch.cat(z_list).detach().cpu().numpy()
    # # data_2d = tsne.fit_transform(all_data)
    # fused_y = sum(h_list_1) / len(h_list_1)
    # data_2d = tsne.fit_transform(fused_y.detach().cpu().numpy())
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.scatter(data_2d[:, 0], data_2d[:, 1], c=predict, label="t-SNE")
    # plt.legend()
    # plt.savefig(f"{args.dataset_name}_{t}.png", dpi=120)
    # plt.show()
    return acc, nmi, ari, pur


def main(args):
    for t in range(1, args.test_times + 1):
        print(f'Test {t}')
        np.random.seed(t)
        random.seed(t)
        # s = 1
        # if(args.missing_rate in (0.7,0.5)):
        #     s = args.seed1
        # elif(args.missing_rate in (0.3,0.1)):
        #     s = args.seed2
        # np.random.seed(s)
        # random.seed(s)
        cmv_data, imv_dataset, sv_datasets = build_dataset(args)
        if(args.missing_rate in (0.1, 0.3)):
            setup_seed(args.seed)
        elif args.missing_rate == 0.5:
            setup_seed(args.seed0)
        else:
            setup_seed(args.seed1)
        # 修改DataLoader配置
        imv_loader = DataLoader(
            imv_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Windows必须为0
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False  # 避免worker持久化
        )
        # imv_loader = DataLoader(imv_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        sv_loaders = [DataLoader(sv_dataset, batch_size=args.batch_size, shuffle=True) for sv_dataset in sv_datasets]
        model = DVIMC(args).to(args.device)

        optimizer = optim.Adam(
            [{"params": model.encoders.parameters(), 'lr': args.learning_rate},
             {"params": model.decoders.parameters(), 'lr': args.learning_rate},
             {"params": model.prior_weight, 'lr': args.prior_learning_rate},
             {"params": model.prior_mu, 'lr': args.prior_learning_rate},
             {"params": model.prior_var, 'lr': args.prior_learning_rate},
             ])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor)
        initialization(model, sv_loaders, cmv_data, args)
        acc, nmi, ari, pur = train(model, optimizer, scheduler, imv_loader, args, t)
        test_record["ACC"].append(acc)
        test_record["NMI"].append(nmi)
        test_record["ARI"].append(ari)
        test_record["PUR"].append(pur)
    print('FINAL_RESULTS: Average ACC {:.2f} Average NMI {:.2f} Average ARI {:.2f} Average PUR {:.2f}'.format(np.mean(test_record["ACC"]) * 100,
                                                                                               np.mean(test_record["NMI"]) * 100,
                                                                                               np.mean(test_record["ARI"]) * 100,
                                                                                               np.mean(test_record["PUR"]) * 100))
    logging.info('Average ACC {:.2f} Average NMI {:.2f} Average ARI {:.2f} Average PUR {:.2f}'.format(np.mean(test_record["ACC"]) * 100,
                                                                                               np.mean(test_record["NMI"]) * 100,
                                                                                               np.mean(test_record["ARI"]) * 100,
                                                                                               np.mean(test_record["PUR"]) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # epochs初始为300 ,learning_rate初始为 0.0005
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--initial_epochs', type=int, default=200, help='initialization epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--prior_learning_rate', type=float, default=0.05, help='initial mixture-of-gaussian learning rate')
    parser.add_argument('--z_dim', type=int, default=10, help='latent dimensions')
    parser.add_argument('--lr_decay_step', type=float, default=10, help='StepLr_Step_size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.9, help='StepLr_Gamma')

    # 在此处设置要读取的数据集
    parser.add_argument('--dataset', type=int, default=2, choices=range(12), help='0:Caltech7-5v, 1:Scene-15, 2:Multi-Fashion, 3:NoisyMNIST, 4:LandUse-21, 5:HandWritten')
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--test_times', type=int, default=2) # 10
    parser.add_argument('--missing_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=5)
    parser.add_argument("--neighbor", default=5, type=int) # 设置邻居的数量
    # 设置隐藏层的大小,原为512，encoder中可能为500
    parser.add_argument("--hidden_size", default=500, type=int) # 设置隐藏层的大小
    # 添加新的超参数
    parser.add_argument('--lamb0', type=float, default=1) # 簇对比超参数
    parser.add_argument('--lamb1', type=float, default=1) # 实例对比超参数
    parser.add_argument('--lamb2', type=float, default=1) # 置信度
    parser.add_argument('--lamb3', type=float, default=1) # Ins
    # 设置np种子
    parser.add_argument('--seed0', type=float, default=1)  # 缺失率0.5的种子
    parser.add_argument('--seed1', type=float, default=1)  # 缺失率0.7的种子
    # parser.add_argument('--seed2', type=float, default=1)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dataset_dir_base = "./npz_data/"

    if args.dataset == 0:  # # test测试集-----------------1
        args.dataset_name = 'Caltech7-5V'
        args.alpha = 5
        args.lamb0 = 1   # 1, 0.1
        args.lamb3 = 1 # 1
        # args.lamb2 = 3   
        args.test_times = 4
        # args.epochs = 300
        args.seed = 16 # 16,13,12(90.71,89.71)
        args.seed0 = 12 # --0.5--
        args.seed1 = 13  # 13: 86.61  损失率：0.7 (5: 85.59)
        args.likelihood = 'Gaussian'
    elif args.dataset == 1:  # # test测试集-----------------2
        args.dataset_name = 'Multi-Fashion'
        args.alpha = 10
        args.lamb0 = 10 # 10,8
        args.lamb3 = 3 # 1 
# ACC 93.59 NMI 91.90 ARI 89.31 PUR 93.89
# ACC 91.80 NMI 89.00 ARI 85.93 PUR 91.99
        args.seed = 9 # --9--(93.59,91.80)--15--(90.59,94.65)
# ACC 89.42 NMI 86.10 ARI 82.28 PUR 89.51
        args.seed0 = 3 # --0.5--(89.42)
# ACC 88.37 NMI 84.59 ARI 80.46 PUR 88.42
        args.seed1 = 4
        
        # args.lamb1 = 8
        args.test_times = 4
        # args.epochs = 300
        args.likelihood = 'Bernoulli'
    # elif args.dataset ==2:
    #     args.dataset_name = 'NoisyMNIST'
    #     args.alpha = 10
    #     args.seed = 10
    #     # args.lamb0 = 4
    #     args.likelihood = 'Bernoulli'
    elif args.dataset ==2:   # test 1 目前可用数据集 # # test测试集 3
        args.dataset_name = 'UCI_Digits' 
        args.alpha = 4  # 6,7
        args.seed = 7 # --7-- 4,5 损失率：0.1,0.3
        args.seed0 = 7 # --0.5--
        args.seed1 = 3 # 3  ,  损失率：0.7
        args.lamb0 = 6  # 6,5,11,10 (6)
        args.lamb3 = 2 # 1,2 (2)
        args.test_times = 1
        # args.epochs = 400
        args.likelihood = 'Gaussian'
    elif args.dataset ==3:   # # test测试集-----------------4
        args.dataset_name = 'HandWritten'
        args.alpha = 10 # 10
        args.seed = 10 # --10--5--,15 损失率：0.1,0.3
        args.seed0 = 10 # --0.5--
        args.seed1 = 1 # 1  损失率：0.7
        args.lamb0 = 5  # 5,4 (5)

        args.lamb3 = 1 # 1 (1)
        args.test_times = 4
        # args.epochs = 400
        args.likelihood = 'Gaussian'
    elif args.dataset ==4:
        args.dataset_name = 'MNIST_USPS'
        args.alpha = 5  # 5
        args.seed = 10   # 10
        args.seed0 = 2024  # --96.74--6--96.62

        args.seed1 = 9  # 3--92.99

        args.lamb0 = 5  # 10
        args.lamb3 = 8   # 3
        args.test_times = 4
        # args.epochs = 300
        args.likelihood = 'Bernoulli'
    elif args.dataset ==5:
        args.dataset_name = 'NoisyMNIST'
        args.alpha = 10
        args.seed = 10
        args.seed0 = 10
        args.seed1 = 10
        args.lamb0 = 4  # 10
        args.lamb3 = 3   # 3
        args.test_times = 4
        args.likelihood = 'Bernoulli'
    elif args.dataset == 8:   # # test测试集
        args.dataset_name = 'Scene-15'
        args.alpha = 20
        args.lamb0 = 1  #
        args.lamb3 = 1  #
        args.test_times = 4
        args.seed = 19     
        args.likelihood = 'Gaussian'
    else:
        args.dataset_name = 'LandUse-21'  # # test测试集
        args.alpha = 11  # 11
        args.seed = 19   #
        # args.lamb0 = 5  # 5,6
        # args.lamb1 = 2 #
        # args.test_times = 10
        # args.epochs = 300
        args.likelihood = 'Gaussian'
    # __________________________________________________ 
    import logging
    log_file = f"log/_Test——{args.dataset_name}_.log"
    # 配置日志
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.info(f"-------------数据集：{args.dataset_name}--------------")
    # logging.info(f"-----seed1：{args.seed1}-----")
    # 初始为  [0.7, 0.5, 0.3, 0.1] # , 0.3, 0.5, 0.7
    for missing_rate in [0.3]: # 0.1, 0.3, 0.5, 0.7
        args.missing_rate = missing_rate
        logging.info(f"Dataset : {args.dataset_name:<15} Missing rate : {args.missing_rate}")
        print(f"Dataset : {args.dataset_name:<15} Missing rate : {args.missing_rate}")
        # for args.lamb3 in [0.001, 0.01, 0.1,1, 5]: # ,10,15,20,25
            # logging.info(f"lamb3 : {args.lamb3}")
            # print(f"lamb3 : {args.lamb3}")
        test_record = {"ACC": [], "NMI": [], "PUR": [], "ARI": []}
        main(args)
    # _________________________________________________

    # # test
    # # 初始为  [0.7, 0.5, 0.3, 0.1]
    # # 设置日志文件路径
    # import logging
    # # log_file = f"log/_卷积_{args.dataset_name}_超参数测试_results.log"
    # log_file = f"log/_lamb3_{args.dataset_name}_.log"
    # # 配置日志
    # logging.basicConfig(filename=log_file, level=logging.INFO, format='%(levelname)s - %(message)s')
    # # logging.info(f"-------------seed={args.seed}--------------")
    # # logging.info(f"-------------消融实验 alpha={args.alpha}--------------")

    # # 设置参数lamb的取值范围，从1到15，步长为1
    # lamb_values = np.arange(11, 21, 1)  # 结束值设置为16以包含15
    
    # # 打印或处理每个lamb值
    # for lamb in lamb_values:
    #     args.lamb3 = lamb
    #     logging.info('------------start-------------')
    #     logging.info(f"args.lamb3={args.lamb3}")
    #     logging.info('------------start-------------')
    #     for missing_rate in [0.1, 0.3, 0.5, 0.7]:
    #         args.missing_rate = missing_rate
    #         # 模拟训练过程
    #         logging.info(f"Dataset : {args.dataset_name:<15} Missing rate : {args.missing_rate}")
    #         print(f"Dataset : {args.dataset_name:<15} Missing rate : {args.missing_rate}")
    #         test_record = {"ACC": [], "NMI": [], "PUR": [], "ARI": []}
    #         main(args)
    # logging.info('-------------end------------')