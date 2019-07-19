# Author:Zhangbingbin 
# Time:2019/7/15
import math
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

# 数据预处理
# 切分文本,得到数据集
def loadDataSet(file):
    df = pd.read_table(file)
    textList = df.Phrase.map(lambda x: x.lower()).str.split('\W+')
    classVec = df['Sentiment']
    return textList, classVec

# textList, classVec = loadDataSet('train.tsv')
# print(textList, classVec)

# 根据词频选择部分词汇作为vocabulary
def createVocabList(textList):
#     vocabSet0 = set([])
    vocabListSet = {}
    for document in textList:
#         vocabSet0 = vocabSet0 | set(document)
        for word in document:
            if word not in vocabListSet.keys():
                vocabListSet[word] = 1
            else:
                vocabListSet[word] += 1
    sorted_vocabListSet = sorted(vocabListSet.items(),key=lambda x:x[1],reverse=True)
    stwlist = [line.strip() for line in open('stopwords.txt', encoding='utf-8').readlines()]
    sorted_vocabList = []
    for i in range(len(sorted_vocabListSet)):
        sorted_vocabList.append(sorted_vocabListSet[i][0])
    vocabList = []
    for word in sorted_vocabList:
        if word not in stwlist and vocabListSet[word] >800 and len(word)>1:
            vocabList.append(word)
    return vocabList

# vocabList = createVocabList(textList)
# print(vocabList)
# print(len(vocabList))

# one-hot编码
def SetOfWords2Vec(vocabList,textList):
    returnVec = []
    for i in  range(len(textList)):
        returnVeci = [0]*len(vocabList)
        for word in textList.iloc[i]:
            if word in vocabList:
                returnVeci[vocabList.index(word)] = 1
        returnVec.append(returnVeci)
    return returnVec

# returnVec = SetOfWords2Vec(vocabList,textList
# print(np.array(returnVec).shape)

# 建立模型
# 定义softmax函数(梯度下降优化w)
class Softmax(object):
    def __init__(self):
        self.learning_rate = 0.00001  # 学习速率
        self.max_iteration = 150000  # 最大迭代次数
        self.weight_lambda = 0.01  # 衰减权重

    def cal_e(self, x, l):
        '''softmax分子'''
        theta_l = self.w[l]
        product = np.dot(theta_l, x)
        return math.exp(product)

    def cal_probability(self, x, j):
        '''softmax'''
        molecule = self.cal_e(x, j)
        denominator = sum([self.cal_e(x, i) for i in range(self.k)])
        return molecule / denominator

    def cal_partial_derivative(self, x, y, j):
        '''计算theta j 的梯度'''
        first = int(y == j)  # 计算示性函数
        second = self.cal_probability(x, j)  # 计算后面那个概率
        return -x * (first - second) + self.weight_lambda * self.w[j]

    def train(self, features, labels):
        self.k = len(set(labels))
        self.w = np.zeros((self.k, len(features[0]) + 1))  # w的维度：k*特征个数+1
        time = 0
        while time < self.max_iteration:
            print('loop %d' % time)
            time += 1
            index = random.randint(0, len(labels) - 1)  # 产生一个随机数
            x = features[index]
            y = labels[index]
            x = list(x)
            x.append(1.0)
            x = np.array(x)
            derivatives = [self.cal_partial_derivative(x, y, j) for j in range(self.k)]
            for j in range(self.k):
                self.w[j] -= self.learning_rate * derivatives[j]  # 梯度下降
            # cost = Cost_function(self.w,np.array(train_features),train_labels_new_labels,self.weight_lambda)
            # print("cost: " + cost)
        return self.w

# 在数据集后面加一列 1 （为计算需要 bias）
def feature_plus(feature):
    feature_plus = []
    for i in range(len(feature)):
        feat = feature[i]
        feat.append(1.0)
        feature_plus.append(feat)
    return feature_plus

# 预测
# 输出预测的标签
def predict_(w,x):
    result = np.dot(x,np.transpose(w))
    arr = np.array(result)
    data = pd.DataFrame(arr, columns=[0, 1, 2, 3, 4])
    pre_labels = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data.values[i, j] == data.max(1).values[i]:
                pre_labels.append(data.columns[j])
    return pre_labels

if __name__=="__main__":
    textList, classVec = loadDataSet('train.tsv')
    # new_labels = One_hot_encode(classVec)
    vocabList = createVocabList(textList)
    returnVec = SetOfWords2Vec(vocabList, textList)
    train_features, test_features, train_labels, test_labels = train_test_split(
        returnVec, classVec, test_size=0.3, random_state=666)
    print(returnVec)
    w = Softmax().train(train_features , list(train_labels))
    print(w)
    print("w dim：" + str(w.shape))
    print("test_features dim：" + str(np.array(test_features).shape))
    print("test_labels dim：" + str(np.array(test_labels).shape))
    test_features_1 = feature_plus(test_features) #####
    pre_labels = predict_(w, test_features_1)
    print(set(pre_labels))
    real_labels = list(test_labels.values)  ####
    dis = list(map(lambda x: x[0] - x[1], zip(pre_labels, real_labels)))
    num = dis.count(0)
    p = float(num)/float(len(test_labels)) ###
    print(p)





