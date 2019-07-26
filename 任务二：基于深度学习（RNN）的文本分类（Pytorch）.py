
# coding: utf-8

import math
import pandas as pd
import numpy as np
import random

#切分文本,得到数据集 
def loadDataSet(file):
    df = pd.read_csv(file,sep="\t")
    textList = df.Phrase.map(lambda x: x.lower()).str.split('\W+')
    classVec= df['Sentiment']
    return df,textList,classVec
  
df,textList, classVec = loadDataSet('train.tsv')
df['textlist'] = textList
print(textList,classVec)
print(df['textlist'])

# 根据词频选择部分词汇作为vocabulary
def createVocabList(textList):
    vocabListSet = {}
    for document in textList:
        for word in document:
            if word not in vocabListSet.keys():
                vocabListSet[word] = 1
            else:
                vocabListSet[word] += 1  
    sorted_vocabListSet = sorted(vocabListSet.items(),key=lambda x:x[1],reverse=True)
#     stwlist = [line.strip() for line in open('stopwords.txt', encoding='utf-8').readlines()]
    
    sorted_vocabList = []
    for i in range(len(sorted_vocabListSet)):
        sorted_vocabList.append(sorted_vocabListSet[i][0])

    vocabList = []
    for word in sorted_vocabList:
            vocabList.append(word)
    return vocabList


vocabList = createVocabList(textList)
print(vocabList)
print(len(vocabList))


#将词汇表中单词编序号
word_to_int = {word: i for i, word in enumerate(vocabList, start=1)}
df['int_textlist'] = df['textlist'].apply(lambda l: [word_to_int[word] for word in l])
df['int_textlist']

#取最长句子长度
max_len = df['int_textlist'].str.len().max()
print(max_len)

#将切分后的文本列表转换位索引列表
all_tokens = np.array([t for t in df['int_textlist']])
encoded_labels = np.array([l for l in df['Sentiment']])
features = np.zeros((len(all_tokens), max_len), dtype=int)
# for each phrase, add zeros at the end 
for i, row in enumerate(all_tokens):
    features[i, :len(row)] = row
print(features[:3])

#划分训练-测试集
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(
        features, encoded_labels, test_size=0.3, random_state=2)

print("train_features dim: "+ str(np.array(train_features).shape))
print("test_features dim: "+ str(np.array(test_features).shape))


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 创建tensor data 
train_data = TensorDataset(torch.LongTensor(train_features), torch.LongTensor(train_labels))
test_data = TensorDataset(torch.LongTensor(test_features), torch.LongTensor(test_labels))
batch_size =34
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)#shuffle:是否将数据打乱
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
# Check the size of the loaders (how many batches inside)
print(len(train_loader))
print(len(test_loader))


# checking if GPU is available
train_on_gpu=torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

class SentimentRNN(nn.Module):
    """RNN 文本分类"""
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        # linear
        self.linear = nn.Linear(hidden_dim, output_size)
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = torch.mean(lstm_out, dim=1)  # 针对第1维进行mean 的平均操作
        out = self.dropout(lstm_out)
        out = self.linear(out)        
        return out, hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers * batch_size * hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
    
        return hidden

# Instantiate the model w/ hyperparams
vocab_size = len(word_to_int)+1 # +1 for the 0 padding
output_size = 5       # 我们所需输出的大小，分类数(0,1,2,3,4)
embedding_dim = 100   # embedding_dim：嵌入查找表中的列数;嵌入的大小。
hidden_dim = 256      # hidden_dim：LSTM单元隐藏层中的单元数。通常更大是更好的性能。常用值为128,256,512等
n_layers = 2          # 网络中LSTM层的数量。通常在1-3之间

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)


# loss and optimization functions
lr=0.003   # learning rate
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params
epochs = 3 # 3-4 is approx where I noticed the validation loss stop decreasing
counter = 0
print_every = 100
clip = 5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()
    
net.train()
# train for some number of epochs
for e in range(epochs):
    h = net.init_hidden(batch_size)    # initialize hidden state
    for inputs, labels in train_loader: # batch loop
        counter += 1
        if(train_on_gpu):
            inputs = inputs.cuda()
            labels = labels.cuda()
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        net.zero_grad()  # zero accumulated gradients
        output, h = net(inputs, h)   # get the output from the model
        loss = criterion(output, labels)  # calculate the loss and perform backprop
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()))

# Get test data loss and accuracy
test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:
    
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])
    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
        
    # get predicted outputs
    output, h = net(inputs, h)
    
    # calculate loss
    test_loss = criterion(output, labels)
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class
    _, pred = torch.max(output,1)
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))

test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

