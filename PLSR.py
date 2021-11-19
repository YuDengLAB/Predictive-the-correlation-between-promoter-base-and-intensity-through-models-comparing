import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import math
from sklearn.metrics import r2_score
import random
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

def load_data(filename):
    train_feat = []
    train_id = []
    data = []
    with open(filename, 'r') as f:
        file = f.readlines()
        for h in file:
            line = h.strip().split(',')

            x_l = [math.log(float(line[0]), 10)]
            # print(len(line[1]))
            #对目标值取log对数
            for a in line[1]: #将序列编码
                if a == 'A':
                    x_l.append(6)
                if a == 'G':
                    x_l.append(1)
                if a == 'T':
                    x_l.append(2)
                if a == 'C':
                    x_l.append(3)
                if a == 'B':
                    x_l.append(0)
            if len(x_l) < 77:
                cha = 77 - len(x_l)
                for q in range(cha):
                    x_l.append(0)
            x_l = np.array(x_l)
            data.append(x_l)
    random.shuffle(data)
    for t in data:
        train_feat.append(t[1:])
        train_id.append(t[0])
    train_feat = np.array(train_feat)
    train_id = np.array(train_id)
    return train_feat, train_id


for i in range(50):
    # params = {'n_estimators': 1000, 'max_depth': 2, 'min_samples_split': 2,
    #           'learning_rate': 0.05, 'loss': 'ls'} #GBDT模型参数，可以通过调整参数提升模型性能
    train_feat, train_id = load_data('train0.csv')
    normalized_test_data = (train_feat - np.mean(train_feat) / np.std(train_feat)) #标准化数据
    X_train, X_test, y_train, y_test = train_test_split(normalized_test_data, train_id, test_size=0.1, random_state=0) #分割数据集
    pls2 = PLSRegression(n_components=15)
    pls2.fit(X_train, y_train)
    # clf = ensemble.GradientBoostingRegressor(**params)
    # clf.fit(X_train, y_train)
    pred = pls2.predict(X_test)
    score = r2_score(y_test, pred) #R2相关系数
    print(score)