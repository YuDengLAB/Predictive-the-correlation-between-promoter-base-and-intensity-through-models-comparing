import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import math
from sklearn.metrics import r2_score
import random
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.cross_decomposition import PLSRegression

def load_data(filename):
    train_feat = []
    train_id = []
    data = []
    with open(filename, 'r') as f:
        file = f.readlines()
        for h in file:
            line = h.strip().split(',')
#
            x_l = [math.log(float(line[0]), 10)]
            for a in line[1]: #将序列编码
                if a == 'A':
                    x_l.append(1)
                if a == 'G':
                    x_l.append(2)
                if a == 'T':
                    x_l.append(3)
                if a == 'C':
                    x_l.append(4)
                if a == 'B':
                    x_l.append(0)
            x_l = np.array(x_l)
            data.append(x_l)
    random.shuffle(data)
    for t in data:
        train_feat.append(t[1:])
        train_id.append(t[0])
        #print(train_id)
    train_feat = np.array(train_feat)
    train_id = np.array(train_id)
    return train_feat, train_id
#
for i in range(20):
    train_feat, train_id = load_data('train0930q.csv')
    normalized_test_data = (train_feat - np.mean(train_feat) / np.std(train_feat)) #标准化数据
    X_train, X_test, y_train, y_test = train_test_split(normalized_test_data, train_id, test_size=0.1, random_state=0) #分割数据集

    regr = AdaBoostRegressor(DecisionTreeRegressor())


    regr.fit(X_train, y_train)
    pred = regr.predict(X_test)

    score = r2_score(y_test, pred)  # R2相关系数
    print(score)
    plt.figure()
    plt.scatter(y_test, pred, s=5,c="k", label="boost0930")
    plt.xlabel("test_state0")
    plt.ylabel("pred_state0")
    plt.legend()
    plt.show()


