import numpy as np
import math
from sklearn.metrics import r2_score
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from scipy import optimize
import matplotlib.pyplot as plt


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
            for a in line[1]:
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
    train_feat = np.array(train_feat)
    train_id = np.array(train_id)
    return train_feat, train_id


#
def f_1(x, A, B):
    return A * x + B


for i in range(20):
    train_feat, train_id = load_data('train0930q.csv')
    normalized_test_data = (train_feat - np.mean(train_feat) / np.std(train_feat))
    X_train, X_test, y_train, y_test = train_test_split(normalized_test_data, train_id, test_size=0.1, random_state=0)
    regr = DecisionTreeRegressor(criterion='mae', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0)
    regr.fit(X_train, y_train)

    pred = regr.predict(X_test)
    pred2 = regr.predict(X_train)
    score = r2_score(y_test, pred)
    score2 = r2_score(y_train[:1000], pred2[:1000])
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
