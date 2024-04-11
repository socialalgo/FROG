
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
df = pd.read_csv("cloth/train.csv")
df=df.dropna()
reviews = df['review'].tolist()

# print(Tweet['review'][2001])
# print(Tweet['text'].tolist())
# vecs = bc.encode(Tweet['text'].tolist())
# vecs = np.array(vecs)
# np.save("textvecs.npy",vecs)
# print(bc.encode(['second do it']))
# train_X = torch.from_numpy(reviews.values)
# train_set = Data.TensorDataset(train_X)
# train_loader = Data.DataLoader(dataset=train_set,
#                                batch_size=32,
#                                shuffle=False)
# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

print("loading")
sentence_embedding = []
for i in tqdm(range(len(reviews))):
    text = reviews[i]
    tokenized_text = tokenizer.tokenize(text) #token初始化
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)[:500] #获取词汇表索引
    tokens_tensor = torch.tensor([indexed_tokens])
    outputs = model(tokens_tensor)  # encoded_layers, pooled_output
    hidden_states = outputs[-2]
    # last_layer = outputs[-1]
    second_to_last_layer = hidden_states[-2]
    token_vecs = second_to_last_layer[0]
    # Calculate the average of all input token vectors.
    sentence_embedding.append(torch.mean(token_vecs, dim=0).detach().numpy())
emb_matrix = np.array(df.values)
emb_matrix = np.c_[emb_matrix[:,[0,1,3]],np.array(sentence_embedding)]

np.savetxt("cloth/emb.csv", emb_matrix, delimiter=',', fmt = '%s')

