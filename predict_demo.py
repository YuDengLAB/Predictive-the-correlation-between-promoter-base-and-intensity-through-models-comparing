import numpy as np
import random
import pandas as pd
import joblib


def load_data(filename):
    train_feat = []
    data = []
    with open(filename, 'r') as f:
        file = f.readlines()
        for h in file:
            line = h.strip().split(',')
            x_l = []
            for a in line[0]: #将序列编码
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
    train_origin = []
    for t in data:
        if len(t) == 74:
            x_o = []
            for o in t:
                if o == 1:
                    x_o.append('A')
                if o == 2:
                    x_o.append('G')
                if o == 3:
                    x_o.append('T')
                if o == 4:
                    x_o.append('C')
                if o == 0:
                    x_o.append('B')
            train_origin.append(''.join(x_o))
            train_feat.append(t)
        if len(train_feat) == 5000:
            break
    train_feat = np.array(train_feat)
    return train_feat, train_origin


train_feat, train_origin = load_data('predict_file.csv')
normalized_test_data = (train_feat - np.mean(train_feat) / np.std(train_feat))
clf = joblib.load("best.pkl")
pred = clf.predict(normalized_test_data)
result = []
for i in pred:
    r = pow(10, i)
    result.append(round(r, 5))
f_result = pd.DataFrame(data=result)
f_origin = pd.DataFrame(data=train_origin)
f_result.to_csv('result.csv', encoding='utf-8')
f_origin.to_csv('origin.csv', encoding='utf-8')
