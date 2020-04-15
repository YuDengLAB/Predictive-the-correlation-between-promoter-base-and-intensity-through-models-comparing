import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import math
from sklearn.metrics import r2_score
import random
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import Imputer
from  sklearn.model_selection import GridSearchCV
from hyperopt import hp
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.externals import joblib

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


def f_1(x, A, B):
    return A * x + B

if __name__ == '__main__':
    score_list = [0]
    for i in range(10):
        train_feat, train_id = load_data('data.csv')
        normalized_test_data = (train_feat - np.mean(train_feat) / np.std(train_feat)) #标准化数据
        cv_params = {'n_estimators': [10,30,50,100,200, 300], 'learning_rate': [0.1, 0.15, 0.08], 'max_depth': [3, 4, 5, 6, 7]} #设置一些参数用来搜索最优参数
        other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1}
        X_train, X_test, y_train, y_test = train_test_split(normalized_test_data, train_id, test_size=0.1, random_state=0) #分割数据集

        model = xgb.XGBRegressor(**other_params)

        optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
        optimized_GBM.fit(X_train, y_train)
        pred = optimized_GBM.predict(X_test)
        pred2 = optimized_GBM.predict(X_train)
        score = r2_score(y_test, pred)  # R2相关系数
        if score >= max(score_list):
            joblib.dump(optimized_GBM, 'best.pkl')
        score_list.append(score)
        score2 = r2_score(y_train[:1000], pred2[:1000])
        print(score, score2)
        # evalute_result = optimized_GBM.grid_scores_
        # print('每轮迭代运行结果:{0}'.format(evalute_result))
        # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
        # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
        plt.rc('font', family='Times New Roman')
        plt.figure()
        A1, B1 = optimize.curve_fit(f_1, y_test, pred)[0]
        x1 = np.arange(min(y_train), max(y_train), 0.01)
        y1 = A1 * x1 + B1
        plt.plot(x1, y1, c="green")
        plt.title('R2=' + str(round(score, 2)))
        plt.scatter(y_train[:1000], pred2[:1000], s=5, c="pink", marker='o', label="Train")
        plt.scatter(y_test[:200], pred[:200], s=5, c="b", marker='x', label="Test")
        plt.xlim(1, 5)
        plt.ylim(1, 5)
        plt.xlabel('Origin')
        plt.ylabel('Predict')
        plt.legend()
        plt.savefig(str(i) + '.pdf')


