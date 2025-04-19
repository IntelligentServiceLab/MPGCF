# 导入所需的库
import pickle
from distutils.command.config import config

import torch.utils.data as data
from sentence_transformers import SentenceTransformer
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch
import scipy.sparse as sp
import numpy as np
import time
from tqdm import tqdm
import os
import glob
import random
# from MLP_MAIN import MLP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(8080)
# SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

# 检查文件路径是否正确
def check_for_pt_files(file_name,folder_path):
    # 使用 glob 模块匹配文件路径模式
    pt_files = glob.glob(os.path.join(folder_path, f'{file_name}.pt'))
    # 检查是否有匹配的文件
    if pt_files:
      return True
    else:
      return False

def get_bert_emb():
    # flag1 和 flag2 判断之前是否生成过mashup和api的嵌入表示
    flag1 = check_for_pt_files('mashup_descr_emb',"./dataset")
    flag2 = check_for_pt_files('api_descr_emb',"./dataset")
    # print(flag1,flag2)
    # 从CSV文件中加载数据
    mashup_data = pd.read_csv("./dataset/Mashup_desc.csv", encoding='UTF-8', header=0)  # 使用Mashups.csv文件
    api_data = pd.read_csv("./dataset/API_desc.csv", encoding='UTF-8', header=0)  # 使用APIs.csv文件

    # 从数据中提取描述信息列
    mashup_descr = mashup_data['description']
    api_descr = api_data['description']

    # 打印描述数据的形状（行数，列数）
    print("shape of mashup_desc ", mashup_descr.shape)
    print("shape of api_desc ", api_descr.shape)

    mashup_descr_emb = bert_convert_emb(mashup_descr) if flag1==False else torch.load('./dataset/mashup_descr_emb.pt',map_location=device)

    api_descr_emb = bert_convert_emb(api_descr) if flag2==False else torch.load('./dataset/api_descr_emb.pt',map_location=device)
    if not flag1:
        torch.save(mashup_descr_emb,'./dataset/mashup_descr_emb.pt')
    if not flag2:
        torch.save(api_descr_emb,'./dataset/api_descr_emb.pt')
    return mashup_descr_emb, api_descr_emb

def bert_convert_emb(descriptions):
    all_sentence_vectors = model.encode(descriptions)
    all_sentence_vectors = torch.tensor(all_sentence_vectors)
    print(all_sentence_vectors.shape)
    return all_sentence_vectors

def get_lgn_data():
    mashup_id,api_id,interaction = [],[],[]
    with open("./dataset/train.txt", 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            items = list(map(int,items))
            for item in items[1:] :
                mashup_id.append(items[0])
                api_id.append(item)
                interaction.append(1)
    num_users = max(mashup_id) + 1
    num_items = max(api_id) + 1
    interaction_matrix = sp.coo_matrix((interaction, (mashup_id, api_id)), shape=(num_users, num_items))

    return interaction_matrix

def get_test_mapping():
    test_mapping = {}
    with open("./dataset/test.txt", 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            items = list(map(int,items))
            test_mapping[items[0]] = items[1:]
    return test_mapping

def get_train_mapping():
    train_mapping = {}
    with open("./dataset/train.txt", 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            items = list(map(int,items))
            train_mapping[items[0]] = items[1:]
    return train_mapping
# def get_train_mapping():
#     with open("")


def get_interaction_rank(n_items):
    mapping = {}
    head_api = []
    tail_api = []
    for i in range(n_items):
        mapping[i] = 0
    # with open("./dataset/train.txt", 'r') as f:
    #     for line in f:
    #         items = line.strip().split(' ')
    #         items = list(map(int,items))
    #         apis = items[1:]
    #         for api in apis :
    #             if api not in mapping.keys():
    #                 mapping[api] = 0
    #             mapping[api] += 1
    # interaction_rank = []
    # for api in mapping.keys():
    #     interaction_rank.append((api,mapping[api]))
    #
    # interaction_rank = sorted(interaction_rank,key=lambda x:x[1],reverse=True)
    #
    # interval = len(interaction_rank)/10 * 2
    #
    # for i in range(len(interaction_rank)):
    #     if i < interval :
    #         head_api.append(interaction_rank[i][0])
    #     else:
    #         tail_api.append(interaction_rank[i][0])
    #
    # print("head api ", head_api,len(head_api))
    # print("tail_api ",tail_api,len(tail_api))
    #
    # with open("./dataset/tail_api.txt", 'w') as f:
    #     for api in tail_api :
    #         api = str(api)
    #         f.write(api)
    #         f.write('\n')
    #
    # with open("./dataset/head_api.txt", 'w') as f:
    #     for api in head_api:
    #         api = str(api)
    #         f.write(api)
    #         f.write('\n')
    with open("./dataset/head_api.txt", 'r') as f:
        for line in f:
            line = line.strip()
            line = int(line)
            head_api.append(line)
    with open("./dataset/tail_api.txt", 'r') as f:
        for line in f:
            line = line.strip()
            line = int(line)
            tail_api.append(line)
            mapping[line] = 1
    print(mapping)
    return head_api, tail_api, mapping

if __name__ == '__main__':
    # get_bert_emb()
    get_interaction_rank(956)