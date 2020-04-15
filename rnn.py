import tensorflow as tf
import numpy as np
from data_process import load_data
from sklearn.metrics import r2_score


data_feat, data_id = load_data('train0.csv')
rnn_unit = 5    #hidden layer units
input_size = 32
output_size = 1
lr = 0.0001

def get_train_data(batch_size=60, time_step=76, train_begin=0, train_end=2500):#函数传值，
    batch_index = []
    data_train = data_feat[train_begin:train_end]#训练数据开始至结束
    # normalized_train_data = (data_train - np.mean(data_train)) / np.std(data_train)  # 定义标准化语句
    train_x, train_y = [], []
    # 训练集
    for i in range(len(data_train)):
        if i % batch_size == 0:
            batch_index.append(i)
        x = data_train[i]
        y = data_id[i, np.newaxis]
        train_x.append(x)
        train_y.append(y)
        # print (train_y)
    # train_y = np.reshape(train_y, [-1, 76, 1])
    train_x = np.reshape(train_x, [-1, 76, 32])

    batch_index.append(len(data_train))

    return batch_index, train_x, train_y
def get_test_data(time_step=76, test_begin=2500):#函数传值，
    data_test = data_feat[test_begin:]#训练数据开始至结束\
    # normalized_test_data = (data_test - np.mean(data_test)) / np.std(data_test)  # 定义标准化语句
    test_x, test_y = [], []
    l = 3000# 训练集
    for i in range(len(data_test)):#以下即是获取训练集并进行标准化，并返回该函数的返回值
        y = data_id[l, np.newaxis]#最后一列标签为Y，可以说出是要预测的，并与之比较，反向求参
        test_x.append(data_test[i])
        test_y.append(y)
        l += 1
    # test_y = np.reshape(test_y, [-1, 76, 1])
    test_x = np.reshape(test_x, [-1, 76, 32])
    return test_x, test_y

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit], stddev=1, seed=1)),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1], stddev=1, seed=1))
}
biases = {
    'in': tf.Variable(tf.constant(0.01, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.01, shape=[1, ]))
}

def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pr = tf.matmul(output, w_out) + b_out
    pred = tf.nn.dropout(pr, keep_prob)
    return pred, final_states


batch_size = 30
time_step = 76
train_begin = 0
train_end = 3000
test_begin = 3000
X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])#预先定义X,Y占位符
Y = tf.placeholder(tf.float32, shape=[None, output_size])
keep_prob = tf.placeholder(tf.float32)
batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)

pred, _ = lstm(X)
predo = []

for i in range(time_step, batch_size*time_step+1):
    if i % 76 == 0:
        predo.append(pred[i-1])

# print(pred)
loss = tf.losses.huber_loss(Y, predo)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)#定义优化
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)#保存模型
test_x, test_y = get_test_data(time_step, test_begin)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
        for step in range(len(batch_index) - 1):#喂数据
            # print(train_x)
            _, loss_, oo = sess.run([train_op, loss, predo], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                                    Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                                                    keep_prob: 1})
            # print(len(oo))
            # exit()
            pred_ = sess.run([pred], feed_dict={X: test_x,
                                                Y: test_y, keep_prob: 1})

            pred__ = []
            for t in range(time_step, time_step*len(test_y)+1):
                if t % 76 == 0:
                    pred__.append(pred_[0][t-1])
        # print(len(pred__))
        # print(len(test_y))
        # test_y_ = []
        # for p in test_y:
        #     for k in p:
        #         test_y_.append(float(k))
        #
        # for q in pred_:
        #     for t in q:
        #         pred__.append(float(t))
        #
            score = r2_score(test_y, pred__)
        # if i % 100 == 0:
        #     print(pred__)
        #     print(test_y_)

            print("Number of iterations:", i, " loss:", loss_, "score:", score)
    print("The train has finished")



