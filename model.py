from xml.sax.handler import all_properties

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from jinja2.nodes import Neg

from utils import BPRLoss,EmbLoss,ndcg_at_k,dcg_at_k
class MOGCL(nn.Module):
    def __init__(self, n_users, n_items, latent_dim, n_layers, str_loss_temp, sem_loss_temp , lambda1,lambda2,lambda3,  r, str_loss_user_weight,str_loss_item_weight, alpha,
                 interaction_matrix, device,sem_loss_item_weight,sem_loss_user_weight,mashup_des,api_des,test_mapping,train_mapping,topk,gama,epoch_num):
        super(MOGCL, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.sem_loss_temp = sem_loss_temp
        self.str_loss_temp = str_loss_temp
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.str_loss_temp = str_loss_temp
        self.sem_loss_temp = sem_loss_temp
        self.sem_loss_user_weight = sem_loss_user_weight
        self.sem_loss_item_weight = sem_loss_item_weight
        self.r = r
        self.str_loss_user_weight = str_loss_user_weight
        self.str_loss_item_weight = str_loss_item_weight
        self.device = device
        self.alpha = torch.tensor(alpha).to(self.device)
        self.mf_loss = BPRLoss(epoch_num)
        self.reg_loss = EmbLoss()
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.item_embedding = nn.Embedding(n_items, latent_dim)
        self.interaction_matrix = interaction_matrix
        self.acc_norm_adj_mat = self.acc_get_norm_adj_mat().to(device)
        self.nacc_norm_adj_mat = self.nacc_get_norm_adj_mat().to(device)
        self.mashup_des = mashup_des
        self.api_des = api_des
        self.init_weights()
        self.test_mapping = test_mapping
        self.train_mapping = train_mapping
        self.top_k = topk

    def init_weights(self):
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

    def acc_get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = inter_M.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D

        L = sp.coo_matrix(L)
        row, col, data = L.row, L.col, L.data
        SparseL = torch.sparse_coo_tensor(torch.LongTensor([row, col]), torch.FloatTensor(data), torch.Size(L.shape))
        return SparseL

    def nacc_get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = inter_M.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        sumArr_left = (A > 0).sum(axis=1)
        diag_left = np.array(sumArr_left.flatten())[0] + 1e-7
        diag_left = np.power(diag_left, -self.r)
        sumArr_right = (A > 0).sum(axis=1)
        diag_right = np.array(sumArr_right.flatten())[0] + 1e-7
        diag_right = np.power(diag_right, -(1 - self.r))

        D_left = sp.diags(diag_left)
        D_right = sp.diags(diag_right)
        L = D_left @ A @ D_right

        L = sp.coo_matrix(L)
        row, col, data = L.row, L.col, L.data
        SparseL = torch.sparse_coo_tensor(torch.LongTensor([row, col]), torch.FloatTensor(data), torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return torch.cat([user_embeddings, item_embeddings], dim=0)

    def forward(self):
        acc_all_embeddings = self.get_ego_embeddings()
        acc_embeddings_list = [acc_all_embeddings]
        nacc_all_embeddings = self.get_ego_embeddings()
        nacc_embeddings_list = [nacc_all_embeddings]

        for _ in range(self.n_layers):
            acc_all_embeddings = torch.sparse.mm(self.acc_norm_adj_mat, acc_all_embeddings)
            nacc_all_embeddings = torch.sparse.mm(self.nacc_norm_adj_mat, nacc_all_embeddings)
            acc_embeddings_list.append(acc_all_embeddings)
            nacc_embeddings_list.append(nacc_all_embeddings)

        lightgcn_acc_all_embeddings = torch.stack(acc_embeddings_list[:self.n_layers + 1],
                                                  dim=1)  # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        lightgcn_acc_all_embeddings = torch.mean(lightgcn_acc_all_embeddings, dim=1)

        user_acc_all_embeddings, item_acc_all_embeddings = torch.split(lightgcn_acc_all_embeddings,
                                                                       [self.n_users, self.n_items])

        lightgcn_nacc_all_embeddings = torch.stack(nacc_embeddings_list[:self.n_layers + 1],
                                                   dim=1)
        lightgcn_nacc_all_embeddings = torch.mean(lightgcn_nacc_all_embeddings, dim=1)

        user_nacc_all_embeddings, item_nacc_all_embeddings = torch.split(lightgcn_nacc_all_embeddings,
                                                                         [self.n_users, self.n_items])
        user_fil_emb = self.alpha * user_acc_all_embeddings + (1 - self.alpha) * user_nacc_all_embeddings
        item_fil_emb = self.alpha * item_acc_all_embeddings + (1 - self.alpha) * item_nacc_all_embeddings


        return lightgcn_acc_all_embeddings, lightgcn_nacc_all_embeddings, user_fil_emb, item_fil_emb

    def str_loss(self, acc_embedding, nacc_embedding, user, item):
        # 计算结构层面的对比学习
        acc_user_embeddings, acc_item_embeddings = torch.split(acc_embedding, [self.n_users, self.n_items])
        nacc_user_embeddings_all, nacc_item_embeddings_all = torch.split(nacc_embedding,
                                                                         [self.n_users, self.n_items])

        acc_user_embeddings = acc_user_embeddings[user]
        nacc_user_embeddings = nacc_user_embeddings_all[user]

        norm_user_emb1 = F.normalize(acc_user_embeddings)
        norm_user_emb2 = F.normalize(nacc_user_embeddings)
        norm_all_user_emb = F.normalize(nacc_user_embeddings_all)
        # 每一个user在acc和nacc的嵌入作为正样本
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)

        # 一个用户的acc嵌入和其余所有用户的nacc嵌入作为负样本
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.str_loss_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.str_loss_temp).sum(dim=1)

        str_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        acc_item_embeddings = acc_item_embeddings[item]
        nacc_item_embeddings = nacc_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(acc_item_embeddings)
        norm_item_emb2 = F.normalize(nacc_item_embeddings)
        norm_all_item_emb = F.normalize(nacc_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.str_loss_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.str_loss_temp).sum(dim=1)

        str_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        str_loss = self.lambda1* (self.str_loss_user_weight * str_loss_user + self.str_loss_item_weight * str_loss_item)
        return str_loss

    def sem_loss(self,uids_des_emb_all,item_des_emb_all,uids_emb_all,item_emb_all,user,item): #语义层面的对比学习

        num_mashup = user.shape[0]
        num_api = item.shape[0]

        uids_des_emb = uids_des_emb_all[user]
        item_des_emb = item_des_emb_all[item]
        norm_user_des_emb = F.normalize(uids_des_emb)
        norm_all_user_des_emb = F.normalize(uids_des_emb_all)
        norm_item_des_emb = F.normalize(item_des_emb)
        norm_all_item_des_emb = F.normalize(item_des_emb_all)

        uids_emb = uids_emb_all[user]
        item_emb = item_emb_all[item]
        norm_user_emb = F.normalize(uids_emb)
        norm_all_user_emb = F.normalize(uids_emb_all)
        norm_item_emb = F.normalize(item_emb)
        norm_all_item_emb = F.normalize(item_emb_all)

        # 用户级的语义对比学习
        pos_score_user = torch.mul(norm_user_emb,norm_user_des_emb).sum(dim=1)
        ttl_score1_user = torch.matmul(norm_user_emb,norm_all_user_des_emb.transpose(0, 1))
        ttl_score2_user = torch.matmul(norm_user_des_emb ,norm_all_user_emb.transpose(0,1) )
        pos_score_user = torch.exp(pos_score_user / self.sem_loss_temp)
        ttl_score1_user = torch.exp(ttl_score1_user / self.sem_loss_temp).sum(dim=1)
        ttl_score2_user = torch.exp(ttl_score2_user / self.sem_loss_temp).sum(dim=1)
        sem_loss_user = -1 / num_mashup * torch.log( ( 2 * pos_score_user ) / (ttl_score1_user + ttl_score2_user)).sum()

        # 项目级的语义对比学习
        pos_score_item = torch.mul(norm_item_emb,norm_item_des_emb).sum(dim=1)
        ttl_score1_item = torch.matmul(norm_item_emb,norm_all_item_des_emb.transpose(0, 1))
        ttl_score2_item = torch.matmul(norm_item_des_emb,norm_all_item_emb.transpose(0,1))
        pos_score_item = torch.exp(pos_score_item / self.sem_loss_temp)
        ttl_score1_item = torch.exp(ttl_score1_item / self.sem_loss_temp).sum(dim=1)
        ttl_score2_item = torch.exp(ttl_score2_item / self.sem_loss_temp).sum(dim=1)
        sem_loss_item = -1 / num_api * torch.log( ( 2 * pos_score_item )  / (ttl_score1_item + ttl_score2_item)).sum()
        sem_loss = self.lambda2 * (self.sem_loss_user_weight * sem_loss_user + self.sem_loss_item_weight * sem_loss_item)
        return sem_loss


    def calculate_loss(self, user,pos,neg1,neg2,epoch,head_api,tail_api):
        # clear the storage variable when training
        pos_item = pos
        neg1_item = neg1
        neg2_item = neg2
        acc_all_embeddings, nacc_all_embeddings, user_all_embeddings, item_all_embeddings = self.forward()

        str_loss = self.str_loss(acc_all_embeddings, nacc_all_embeddings, user, pos_item)

        sem_loss = self.sem_loss(self.mashup_des, self.api_des,user_all_embeddings, item_all_embeddings,user,pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg1_embeddings = item_all_embeddings[neg1_item]
        neg2_embeddings = item_all_embeddings[neg2_item]

        uids_des_emb = self.mashup_des[user]
        pos_des_emb = self.api_des[pos_item]
        neg1_des_emb = self.api_des[neg1_item]
        neg2_des_emb = self.api_des[neg2_item]

        uids_des_emb = uids_des_emb / uids_des_emb.norm(p=2,dim=-1,keepdim=True)
        pos_des_emb = pos_des_emb / pos_des_emb.norm(p=2,dim=-1,keepdim=True)
        neg1_des_emb = neg1_des_emb / neg1_des_emb.norm(p=2,dim=-1,keepdim=True)
        neg2_des_emb = neg2_des_emb / neg2_des_emb.norm(p=2, dim=-1, keepdim=True)

        #将文本嵌入与结构嵌入融合
        final_mashup_emb = (u_embeddings + uids_des_emb) / 2
        final_pos_emb = (pos_embeddings + pos_des_emb) / 2
        final_neg1_emb = (neg1_embeddings + neg1_des_emb) / 2
        final_neg2_emb = (neg2_embeddings + neg2_des_emb) / 2



        # calculate BPR Loss
        #改进的bpr loss 每个正样本和两个负样本进行对照，一个负样本来自头部项目，另一个样本来自尾部项目
        #使用点积来代表两个嵌入的相似程度
        # user_list = user.tolist()
        # pos_list = pos.tolist()
        # bpr_loss_h = 0
        # bpr_loss_t = 0
        # for i in range(len(user_list)):
        #     u = user_list[i]
        #     u_des_emb = self.mashup_des[u]
        #     u_des_emb = u_des_emb / u_des_emb.norm(p=2, dim=-1, keepdim=True)
        #     u_emb = (user_all_embeddings[u] + u_des_emb) / 2
        #     Pos = pos_list[i]
        #     Pos_des_emb = self.api_des[Pos]
        #     Pos_des_emb = Pos_des_emb / Pos_des_emb.norm(p=2, dim=-1, keepdim=True)
        #     Pos_emb = (item_all_embeddings[Pos] + Pos_des_emb) / 2

            # for h_api in head_api:
            #     if h_api not in self.test_mapping[user_list[i]]: #说明是负样本
            #         neg_des_emb = self.api_des[h_api]
            #         neg_des_emb = neg_des_emb / neg_des_emb.norm(p=2, dim=-1, keepdim=True)
            #         neg_emb = (item_all_embeddings[h_api] + neg_des_emb) / 2
            #         weight = torch.mul(Pos_emb, neg_emb).sum(dim=1)
            #         pos_score = torch.mul(u_emb, Pos_emb).sum(dim=1)
            #         neg_score = torch.mul(u_emb, neg_emb).sum(dim=1)
            #         bpr_loss_h += -torch.log(torch.sigmoid(pos_score - weight * neg_score))
            # for t_api in tail_api:
            #     if t_api not in self.test_mapping[user_list[i]]: #说明是负样本
            #         neg_des_emb = self.api_des[t_api]
            #         neg_des_emb = neg_des_emb / neg_des_emb.norm(p=2, dim=-1, keepdim=True)
            #         neg_emb = (item_all_embeddings[t_api] + neg_des_emb) / 2
            #         weight = torch.mul(Pos_emb, neg_emb).sum(dim=1)
            #         pos_score = torch.mul(u_emb, Pos_emb).sum(dim=1)
            #         neg_score = torch.mul(u_emb, neg_emb).sum(dim=1)
            #         bpr_loss_h += -torch.log(torch.sigmoid(pos_score - weight * neg_score))



        # pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        # neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)





        pos_scores = torch.mul(final_mashup_emb, final_pos_emb).sum(dim=1)
        neg1_scores = torch.mul(final_mashup_emb, final_neg1_emb).sum(dim=1)
        neg2_scores = torch.mul(final_mashup_emb, final_neg2_emb).sum(dim=1)
        neg1_weight = torch.mul(final_pos_emb, final_neg1_emb).sum(dim=1)
        neg2_weight = torch.mul(final_pos_emb, final_neg2_emb).sum(dim=1)
        neg1_scores = neg1_weight * neg1_scores
        neg2_scores = neg2_weight * neg2_scores
        #

        mf_loss = self.mf_loss(pos_scores, neg1_scores, neg2_scores,epoch)


        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg1_ego_embeddings = self.item_embedding(neg1_item)
        neg2_ego_embeddings = self.item_embedding(neg2_item)


        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg1_ego_embeddings, neg2_ego_embeddings)
        return mf_loss + self.lambda3 * reg_loss, str_loss , sem_loss


    def pred(self,uids,is_tail_api):
        topk = self.top_k
        acc_all_embeddings, nacc_all_embeddings, user_all_embeddings, item_all_embeddings = self.forward()
        uids_des_emb = self.mashup_des[uids]
        item_des_emb = self.api_des

        uids_des_emb = uids_des_emb / uids_des_emb.norm(p=2,dim=-1,keepdim=True)
        item_des_emb = item_des_emb / item_des_emb.norm(p=2,dim=-1,keepdim=True)

        uids_emb = user_all_embeddings[uids]
        item_emb = item_all_embeddings

        final_mashup_emb = (uids_des_emb + uids_emb) / 2
        final_api_emb = (item_des_emb + item_emb) / 2

        # final_mashup_emb = uids_emb
        # final_api_emb = item_emb


        scores = torch.matmul(final_mashup_emb, final_api_emb.T)  # (batch_size, num_items)
        all_recall = 0
        all_ndcg = 0
        all_tail = 0
        coverage_api = set()
        for i in range(len(uids)):
            real_api = torch.tensor(self.train_mapping[uids[i]], dtype=torch.long)  # 确保是 LongTensor
            scores[i, real_api] = -float("inf")  # 避免选到已交互项
        topk_values, topk_indices = torch.topk(scores, k=topk, dim=1)
        for i in range(len(uids)):
            hit = 0
            hit_tail = 0
            needy_api = self.test_mapping[uids[i]]
            # print(i,"   ",topk_indices[i],"   ",needy_api)
            recommended_scores = topk_values[i].tolist()
            ground_truth_scores = []
            for pred_api in topk_indices[i]:
                pred_api = pred_api.item()
                coverage_api.add(pred_api)
                if pred_api in needy_api:
                    hit +=1
                    ground_truth_scores.append(1)
                else:
                    ground_truth_scores.append(0)
                if is_tail_api[pred_api]==1:
                    hit_tail += 1

            all_tail += hit_tail/self.top_k
            all_recall += hit/len(needy_api)
            all_ndcg += ndcg_at_k(recommended_scores, ground_truth_scores,topk)
        all_recall = all_recall/len(uids)
        all_ndcg = all_ndcg/len(uids)
        all_tail = all_tail/len(uids)
        return all_recall,all_ndcg,coverage_api,all_tail