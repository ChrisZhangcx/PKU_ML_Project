# -*- coding:utf8 -*-

# 用主成分分析降维
import csv
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np


# 传入需要降维的数据集，以及最终要达到的维度，返回降维后的数据
def pca(x, dim):
    pca = PCA(n_components=dim)
    newx = pca.fit_transform(x)
    print pca.explained_variance_ratio_
    return newx


# 读入原始数据
csvfile = file("X_for_initial.csv", "rb")
reader = csv.reader(csvfile)
X = []
for each in reader:
    tmpX = []
    for i in range(len(each)):
        tmpX.append(int(each[i]))
    X.append(tmpX)
csvfile.close()
X = np.array(X)
# 数据降维
X_after = pca(X, int(0.2*X.shape[1]))
# 保存降维后的数据
csvfile = file("pca_after.csv", "wb")
writer = csv.writer(csvfile)
for each in X_after:
    writer.writerow(each)
csvfile.close()
