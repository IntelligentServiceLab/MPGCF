from model import MOGCL
from data_loader import get_bert_emb,get_lgn_data,get_test_mapping,get_train_mapping,get_interaction_rank
from utils import ModelConfig,MLP,TrnData,EarlyStopping
import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = ModelConfig()
n_users = config.n_users
n_items = config.n_items
latent_dim = config.latent_dim
n_layers = config.n_layers
str_loss_temp = config.str_loss_temp
sem_loss_temp = config.sem_loss_temp
lambda1 = config.lamdba1
lambda2 = config.lamdba2
lambda3 = config.lamdba3
str_loss_user_weight = config.str_loss_user_weight
str_loss_item_weight = config.str_loss_item_weight
alpha = config.alpha
epochs = config.epochs
lr = config.lr
r = config.r
batch_size = config.batch_size
sem_loss_item_weight = config.sem_loss_item_weight
sem_loss_user_weight = config.sem_loss_user_weight
test_batch_size = config.test_batch_size
topk = config.topk
gama = config.gama

if __name__ == '__main__':

    train_mapping = get_train_mapping()
    test_mapping = get_test_mapping()
    head_api, tail_api ,is_tail_api = get_interaction_rank(n_items)
    interaction_matrix = get_lgn_data()
    # interaction_matrix.to(device=device)
    mashup_des_emb,api_des_emb = get_bert_emb()
    mlp = MLP(384,latent_dim,device)
    mlp.to(device)
    mashup_des_emb = mlp(mashup_des_emb)
    api_des_emb = mlp(api_des_emb)
    # mashup_des_emb.to(device=device)
    # api_des_emb.to(device=device

    model = MOGCL(n_users=n_users, n_items=n_items,latent_dim=latent_dim,n_layers=n_layers,str_loss_temp=str_loss_temp,sem_loss_temp=sem_loss_temp,lambda1=lambda1,lambda2=lambda2,lambda3=lambda3,
                  r=r,str_loss_item_weight=str_loss_item_weight,str_loss_user_weight=str_loss_user_weight,alpha=alpha,interaction_matrix=interaction_matrix,device=device,
                  sem_loss_item_weight=sem_loss_item_weight,sem_loss_user_weight=sem_loss_user_weight,mashup_des=mashup_des_emb,api_des=api_des_emb,test_mapping=test_mapping,train_mapping=train_mapping,topk=topk,gama=gama,epoch_num = epochs)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = TrnData(interaction_matrix)

    train_loader = data.DataLoader(train_data,batch_size=batch_size,shuffle=True)

    early_stopping = EarlyStopping(patience=5000, min_delta=0.0001) # 早停机制


    for epoch in tqdm(range(1,epochs),total=epochs):
        train_loader.dataset.neg_sampling(head_api,tail_api)
        for i,batch in enumerate(train_loader):
            uids, pos, neg1, neg2 = batch
            uids = uids.long().cuda(torch.device(device))
            pos = pos.long().cuda(torch.device(device))
            neg1 = neg1.long().cuda(torch.device(device))
            neg2 = neg2.long().cuda(torch.device(device))
            optimizer.zero_grad()
            bpr_loss,cl_loss,sem_loss = model.calculate_loss(uids, pos, neg1 ,neg2 ,epoch,head_api,tail_api)
            total_loss = bpr_loss + cl_loss + sem_loss
            # total_loss = bpr_loss + sem_loss
            # print(total_loss.item())
            total_loss.backward(retain_graph=True)
            optimizer.step()

        # print("epoch=",epoch,"   loss=",epoch_loss)
        if (epoch % 5 == 0) :
            model.eval()
            coverage_api = set()
            test_uids = np.array([i for i in range(interaction_matrix.shape[0])])
            batch_no = int(np.ceil(len(test_uids) / test_batch_size))
            all_recall = 0
            all_ndcg = 0
            all_coverage = 0
            all_tail = 0
            for batch in range(batch_no):
                start = batch * test_batch_size
                end = min((batch + 1) * test_batch_size , len(test_uids))
                recall, ndcg, coverage, tail  = model.pred(test_uids[start:end], is_tail_api)
                all_recall += recall
                all_ndcg += ndcg
                all_tail += tail
                coverage_api.update(coverage)
            all_recall = all_recall/batch_no
            all_ndcg = all_ndcg/batch_no
            all_tail = all_tail/batch_no
            all_coverage = len(coverage_api)/n_items
            print("epoch:",epoch,"   recall:",all_recall,"   ndcg:",all_ndcg,"   coverage:",all_coverage,"   tail",all_tail)
            if early_stopping(all_recall):
                print("Stopping training early!")
                break


