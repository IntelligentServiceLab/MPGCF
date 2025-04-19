import torch.nn as nn
import torch
import torch.utils.data as data
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MLP(nn.Module):
    def __init__(self, input_dim,  out_dim, device):
        super(MLP, self).__init__()  # 调用父类的 __init__ 方法
        self.seq = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
        )
        self.device = device
    def forward(self, x):
        x = self.seq(x)
        x.to(self.device)
        return x


class ModelConfig:
    def __init__(self):
        self.n_users = 2289
        self.n_items = 956

        self.latent_dim = 256

        self.n_layers = 3

        self.lamdba1 = 1e-5 # str_loss_weight
        self.lamdba2 = 1e-3 # sem_loss_weight
        self.lamdba3 = 1e-4 # reg_loss_weight

        self.str_loss_item_weight = 0.7 # str_loos 中项目部分的权重1.5，用户部分的权重是1
        self.str_loss_user_weight = 0.3

        self.str_loss_temp = 0.1
        self.sem_loss_temp = 0.1

        self.sem_loss_user_weight = 1e-3
        self.sem_loss_item_weight = 1e-3

        self.r = 1.25
        self.alpha = 0.7
        self.epochs = 500
        self.lr = 0.0001

        self.batch_size = 4096
        self.test_batch_size = 512

        self.topk = 5

        self.gama = 0.5

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs1 = np.zeros(len(self.rows)).astype(np.int32)
        self.negs2 = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self, head_api , tail_api):  # 需要有两种负样本 ， 一种是来自流行项目的，另一种是来自长尾项目的
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg1 = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg1) not in self.dokmat and i_neg1 in head_api:
                    break
            self.negs1[i] = i_neg1
            while True:
                i_neg2 = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg2) not in self.dokmat and i_neg2 in tail_api:
                    break
            self.negs2[i] = i_neg2

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs1[idx], self.negs2[idx]



class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, k , gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma
        self.maxx = 0.5
        self.minn = 0.5
        self.epoch_num = k

    def forward(self, pos_score, neg1_score, neg2_score,epoch):
        loss_head = -torch.log(torch.sigmoid(pos_score - neg1_score)).mean()
        loss_tail = -torch.log(torch.sigmoid(pos_score - neg2_score)).mean()
        weight = ( 1 - epoch / self.epoch_num ) * self.maxx + epoch / self.epoch_num * self.minn
        return (1-weight) * loss_head + weight * loss_tail


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


def dcg_at_k(scores, k):
    """
    计算 DCG@k
    :param scores: 排序后的相关性分数
    :param k: 前 k 个位置
    :return: DCG 值
    """
    scores = np.asfarray(scores)[:k]
    if scores.size == 0:
        return 0.0
    return np.sum((2 ** scores - 1) / np.log2(np.arange(2, scores.size + 2)))


def ndcg_at_k(predicted_scores, true_scores, k):
    """
    计算 NDCG@k
    :param predicted_scores: 模型预测的分数
    :param true_scores: 实际的相关性分数
    :param k: 评价的前 k 个位置
    :return: NDCG 值
    """
    # 按预测分数排序后的实际分数
    sorted_true_scores = [true for _, true in sorted(zip(predicted_scores, true_scores), reverse=True)]

    # 计算 DCG 和 IDCG
    dcg = dcg_at_k(sorted_true_scores, k)
    idcg = dcg_at_k(sorted(true_scores, reverse=True), k)

    # 避免除以 0 的情况
    return dcg / idcg if idcg > 0 else 0.0

class EarlyStopping:
    # 早停策略，如果模型的性能在连续的几次测试中不再提升，就可以停止训练了

    def __init__(self, patience=3, min_delta=0.0001):

        self.patience = patience
        self.min_delta = min_delta
        self.best_score = 0  # 初始化效果
        self.counter = 0  # 记录连续多少个 epoch 没有改善

    def __call__(self, val_score):
        """ 在每个 epoch 结束后调用，检查是否需要停止训练 """
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0  # 重新计数
        else:
            self.counter += 1  # 连续不改善次数加 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs of no improvement.")
                return True  # 训练应当停止
        return False  # 继续训练



