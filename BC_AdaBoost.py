# -*- coding:utf8 -*-

# 用python实现balacecascade
# 弱学习算法：decision stump
import random
import csv
import numpy as np
import math
import time
from sklearn import cross_validation

"""
函数定义
"""

# 过采样函数：传入数据列表与一次的采样样本数，返回采样结果（列表形式）
def repetitionRandomSampling(dataMat, number):
    sample = []
    for i in range(number):
        sample.append(dataMat[random.randint(0, len(dataMat)-1)])
    return sample


# 选择最好的分类特征与阈值（传入X1,X2,迭代次数si），输出特征向量w与阈值向量t
def adaBoost(X1, X2, si):
    N = X1.shape[1]
    # 初始化特征向量与特征阈值
    w = np.zeros(N)         # 特征向量w：为0表示
    t = np.zeros(N)
    # 初始化样本的权值向量
    W1 = np.ones(len(X1))/(len(X1)+len(X2))
    W2 = np.ones(len(X2))/(len(X2)+len(X2))
    # 寻找最好的特征
    numStep = 10
    # 对弱分类器进行si次迭代
    for i in range(si):
        bestStump = {}
        minErr = np.inf
        # 找到最好的特征与阈值
        for k in range(N):
            rangeMin, rangeMax = X[:, k].min(), X[:, k].max()
            stepSize = (rangeMax - rangeMin)/numStep
            for j in range(-1, numStep+1):
                thresh = rangeMin + j*stepSize
                for ineq in ['lt', 'gt']:
                    # 注：该错误率即为分类误差率em
                    error = stumpClassify(X1, X2, W1, W2, k, thresh, ineq)
                    if minErr > error:
                        minErr = error
                        #print "minErr = " + str(minErr)
                        bestStump['dim'] = k
                        bestStump['ineq'] = ineq
                        bestStump['thresh'] = thresh
                        bestStump['error'] = minErr
        #print bestStump
        # 计算弱分类器的系数alpha - 第si次迭代的系数保存在alpha列表下标为si-1的变量中
        thisAlpha = 0.5*math.log((1-bestStump['error'])/bestStump['error'])
        # 更新所有变量的权值分布
        Zm = 0
        k = bestStump['dim']
        ineq = bestStump['ineq']
        thresh = bestStump['thresh']
        if ineq == 'lt':
            # 判断每个X1是否分类正确，之后加到Zm中，最后更新权值
            for p in range(len(X1)):
                # 分类正确
                if X1[p][k]-thresh < 0:
                    Zm += W1[p]*math.exp(-thisAlpha*1.0)
                    W1[p] *= math.exp(-thisAlpha*1.0)
                else:
                    Zm += W1[p]*math.exp(thisAlpha*1.0)
                    W1[p] *= math.exp(thisAlpha*1.0)
            for p in range(len(X2)):
                # 分类正确
                if X2[p][k]-thresh >= 0:
                    Zm += W2[p]*math.exp(-thisAlpha*1.0)
                    W2[p] *= math.exp(-thisAlpha*1.0)
                else:
                    Zm += W2[p]*math.exp(-thisAlpha*1.0)
                    W2[p] *= math.exp(-thisAlpha*1.0)
        else:
            # 判断每个X1是否分类正确，之后加到Zm中，最后更新权值
            for p in range(len(X1)):
                # 分类正确
                if X1[p][k]-thresh > 0:
                    Zm += W1[p]*math.exp(-thisAlpha*1.0)
                    W1[p] *= math.exp(-thisAlpha*1.0)
                else:
                    Zm += W1[p]*math.exp(thisAlpha*1.0)
                    W1[p] *= math.exp(thisAlpha*1.0)
            for p in range(len(X2)):
                # 分类正确
                if X2[p][k]-thresh <= 0:
                    Zm += W2[p]*math.exp(-thisAlpha*1.0)
                    W2[p] *= math.exp(-thisAlpha*1.0)
                else:
                    Zm += W2[p]*math.exp(thisAlpha*1.0)
                    W2[p] *= math.exp(thisAlpha*1.0)

        for p in range(len(X1)):
            W1[p] = W1[p]/Zm
            W2[p] = W2[p]/Zm
        #print "Zm = "+str(Zm)
        # 保存本次弱分类器结果
        if ineq == 'lt':                # 如果负例<0，正例>0则系数不变
            w[k] += 1.0*thisAlpha
            t[k] += thresh*thisAlpha
        else:                           # 否则系数取反
            w[k] -= 1.0*thisAlpha
            t[k] -= thresh*thisAlpha
    # 返回最终分类器
    return w, t


# 分类树根：传入数据集，属性列，阈值与比较方式，根据样本的权值分配，计算错误指数
def stumpClassify(X1, X2, W1, W2, j, thresh, ineq):
    error = 0
    # 当条件为小于'lt'时，所有该特征小于阈值的都认为是负例
    if ineq == 'lt':
        for i in range(0, len(X1)):
            if X1[i][j] >= thresh:
                error = error + W1[i]
        for i in range(0, len(X2)):
            if X2[i][j] < thresh:
                error = error + W2[i]
    # 当条件为大于'gt'时，所有该特征大于阈值的都认为是负例
    else:
        for i in range(0, len(X1)):
            if X1[i][j] <= thresh:
                error = error + W1[i]
        for i in range(0, len(X2)):
            if X2[i][j] > thresh:
                error = error + W2[i]
    return error


# 精度检验函数：传入特征向量，特征阈值与当前负例样本，输出在目前分类器在负例样本中的错误率
def errorrateCalculate(w, t, X1):
    wrong3 = 0
    #csvfile = file("record_x1.csv", 'wb')
    #writer = csv.writer(csvfile)
    outcome = []                # 保存本次分类的所有结果
    for each in X1:
        result = 0
        for q in range(len(w)):
            if w[q] != 0:
                result += (each[q]*w[q]-t[q])
        #writer.writerow([result])
        outcome.append(result)
        if result >= 0:
            wrong3 += 1
    #csvfile.close()
    #print "该分类器在所有负例中的错误数量为：" + str(wrong3)
    #print "该分类器的当前错误率为：" + str(1.0*wrong3/len(X1))
    return 1.0*wrong3/len(X1), outcome


# 最终训练效果检验函数：传入特征向量、特征阈值与正负例样本数据，输出正例精确度、负例精确度与性能度量标准
def finalCheck(weight, thresh, theta, x1, x2):
    # 计算两个阈值的和
    totalThresh = 0
    for i in range(len(thresh)):
        totalThresh += thresh[i]
    print
    totalTheta = 0
    for i in range(len(theta)):
        totalTheta += theta[i]
    # 注意：这里是
    totalTheta *= 0.5
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    wrong1 = 0
    for each in x1:
        # 对负例，当其大于等于0时，即认为分类错误
        if np.dot(each, weight)-totalThresh+totalTheta >= 0:
            wrong1 += 1
            FP += 1
        else:
            TN += 1
    wrong2 = 0
    for each in x2:
        # 对正例，当其小于0时，认为分类错误
        if np.dot(each, weight)-totalThresh+totalTheta < 0:
            wrong2 += 1
            FN += 1
        else:
            TP += 1

    # 计算查准率P
    if TP+FP == 0:
        print "由于预测为正例的人数为0，本次计算F1值为0"
        return 0
    P = 1.0*TP/(TP+FP)
    R = 1.0*TP/(TP+FN)
    F1 = (2.0*P*R)/(P+R)
    print "参加预测总人数：" + str(len(x1)+len(x2))
    print "其中患者总人数：" + str(len(x2))
    print "预测为患者的人数：" + str(TP+FP)
    print "其中真正患者的数目：" + str(TP)
    print "P = " + str(P) + ", R = " + str(R)
    print "F1值为：" + str(F1)
    return F1


"""
正式代码部分
"""
start = time.clock()
X = []
y = []
# 读取数据并保存
csvfile = file("X_for_initial.csv", 'rb')
reader = csv.reader(csvfile)
for line in reader:
    line1 = []
    for content in line:
        line1.append(float(content))
    X.append(line1)
csvfile.close()
csvfile = file("y_for_initial.csv", 'rb')
reader = csv.reader(csvfile)
for line in reader:
    line1 = []
    for content in line:
        line1.append(int(content))
    y.append(line1)
csvfile.close()
X = np.array(X)
y = np.array(y)

# 将未患病与患病人群划分为两个组X1,X2
X1 = []         # X1为未患病人群，其y值为-1
X2 = []         # X2为患病人群，其y值为1
for i in range(len(X)):
    if y[i] == 1:
        X2.append(X[i])
    else:
        X1.append(X[i])
X1 = np.array(X1)
X2 = np.array(X2)
y1 = np.ones(len(X1))
y1 = -1.0*y1
y2 = np.ones(len(X2))

"""
T 迭代次数，指循环次数
si每个弱分类器的迭代次数
i 初始循环次数
f 每次循环保留的正例数占总正例数的比值，保证了T次循环后剩余样本必定少于少数类样本
theta 通过阈值来控制f
"""
# 五次五折交叉验证
finalF1 = 0
finalPeople = 0
for times in range(0, 5):
    # 划分训练集与测试集
    X1_train, X1_test, y1_train, y1_test = cross_validation.train_test_split(X1, y1, test_size=0.2, random_state=0)
    X2_train, X2_test, y2_train, y2_test = cross_validation.train_test_split(X2, y2, test_size=0.2, random_state=0)
    # 计算患者的数目
    patientNumber = len(X2_train)
    # 初始化参数
    T = 10                                              # 循环次数
    si = 5                                              # 每个adaboost分类器的迭代次数
    i = 0
    f = (1.0 * len(X2_train) / len(X1_train)) ** (1.0 / (T - 1))
    theta = []  # 保存每个循环产生分类器的附加阈值
    # balacecascade算法实现
    w = np.zeros(X2.shape[1])
    t = np.zeros(X2.shape[1])
    while i < T:
        i += 1
        resultw = []
        resultt = []

        # 过采样
        if len(X1_train) <= patientNumber:  # 额外判断条件：如果X1中人数已经不到X2人数，则直接退出
            break
        sample = repetitionRandomSampling(X1_train, patientNumber)
        sample = np.array(sample)

        # 学习获得一个由si个弱分类器构成的强分类器（此为一个adaboost过程）
        resultw, resultt = adaBoost(sample, X2_train, si)

        # 将该强分类器保存到结果中
        for q in range(len(w)):
            w[q] += resultw[q]
            t[q] += resultt[q]

        # 求出阈值theta[i-1]满足第i个分类器满足FPR=f
        thisf, outcome = errorrateCalculate(resultw, resultt, X1_train)  # 计算当前的f值
        if math.fabs(thisf - f) > 0.01:  # 取阈值为0.01，即当两者差的绝对值大于阈值时，认为两者准确率是不同的
            thistheta = 0
            if thisf < f:
                # 当当前错误率较低时，为使负例分类错误率增加，应当加上一个正的阈值theta
                while thisf < f:
                    wrong3 = 0
                    for each_result in range(len(outcome)):
                        outcome[each_result] += 0.3
                        if outcome[each_result] > 0:
                            wrong3 += 1
                    thisf = 1.0 * wrong3 / len(X1_train)
                    thistheta += 0.3
                # 回到上一个状态，使错误率略低于f
                wrong3 = 0
                thistheta -= 0.3
                for each_result in range(len(outcome)):
                    outcome[each_result] -= 0.3
                    if outcome[each_result] > 0:
                        wrong3 += 1
                thisf = 1.0 * wrong3 / len(X1_train)
            else:
                # 当当前错误率较高时，为使负例分类错误率减小，应当加上一个负的阈值theta
                while thisf > f:
                    wrong3 = 0
                    for each_result in range(len(outcome)):
                        outcome[each_result] -= 0.3
                        if outcome[each_result] > 0:
                            wrong3 += 1
                    thisf = 1.0 * wrong3 / len(X1_train)
                    thistheta -= 0.3

        # print "为保证负例的分类错误率为" + str(f) + ", 需要加上一个阈值：" + str(thistheta)
        # print "此时的分类错误率为 " + str(thisf)
        theta.append(thistheta)

        # 使用当前得到的训练器进行分类，从X1中去除能够正确分类的样本
        totalt = 0
        for each in resultt:
            totalt += each
        tmpX = []
        for each_sample in range(len(X1_train)):
            result = np.dot(X1_train[each_sample], resultw)
            result -= totalt
            result += thistheta
            if result >= 0:
                tmpX.append(X1_train[each_sample])
        tmpX = np.array(tmpX)
        X1_train = tmpX

    print "本次AdaBoost 模型："
    print w
    print t
    print theta
    finalF1 += finalCheck(w, t, theta, X1_test, X2_test)

print "5次5折交叉验证最终F1值：" + str(finalF1/5.0)
end = time.clock()

print "time = %f s" % (end-start)

