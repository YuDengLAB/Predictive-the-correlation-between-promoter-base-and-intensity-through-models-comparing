import tensorflow as tf
import numpy as np
from data_process import load_data
from sklearn.metrics import r2_score


data_feat, data_id = load_data('train0.csv')
rnn_unit = 5    #hidden layer units
input_size = 32
output_size = 1
lr = 0.0001

def get_train_data(batch_size=60, time_step=76, train_begin=0, train_end=2500):
    batch_index = []
    data_train = data_feat[train_begin:train_end]
    # normalized_train_data = (data_train - np.mean(data_train)) / np.std(data_train)
    train_x, train_y = [], []

    for i in range(len(data_train)):
        if i % batch_size == 0:
            batch_index.append(i)
        x = data_train[i]
        y = data_id[i, np.newaxis]
        train_x.append(x)
        train_y.append(y)
        # print (train_y)
    train_x = np.reshape(train_x, [-1, 76, 32])

    batch_index.append(len(data_train))

    return batch_index, train_x, train_y
def get_test_data(time_step=76, test_begin=2500):
    data_test = data_feat[test_begin:]
    test_x, test_y = [], []
    l = 3000
    for i in range(len(data_test)):
        y = data_id[l, np.newaxis]
        test_x.append(data_test[i])
        test_y.append(y)
        l += 1
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
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
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
X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
Y = tf.placeholder(tf.float32, shape=[None, output_size])
keep_prob = tf.placeholder(tf.float32)
batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)

pred, _ = lstm(X)
predo = []

for i in range(time_step, batch_size*time_step+1):
    if i % 76 == 0:
        predo.append(pred[i-1])

loss = tf.losses.huber_loss(Y, predo)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
test_x, test_y = get_test_data(time_step, test_begin)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        for step in range(len(batch_index) - 1):
            _, loss_, oo = sess.run([train_op, loss, predo], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                                    Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                                                    keep_prob: 1})
            pred_ = sess.run([pred], feed_dict={X: test_x,
                                                Y: test_y, keep_prob: 1})

            pred__ = []
            for t in range(time_step, time_step*len(test_y)+1):
                if t % 76 == 0:
                    pred__.append(pred_[0][t-1])
            score = r2_score(test_y, pred__)

            print("Number of iterations:", i, " loss:", loss_, "score:", score)
    print("The train has finished")




