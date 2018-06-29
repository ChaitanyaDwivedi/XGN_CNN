import pickle
from loadData import dataLoader
import tensorflow as tf
from CalculateEER import SKGetEER as GetEER
import numpy as np

data = pickle.load(open("./Data/dataBenchmarkTf.pickle", "rb"))

X = data['data']
Y = data['lables']
X = X.reshape(20400, 1, 31, 1)

data['data'] = X
pickle.dump(data, open("./Data/cnnkeystroke.pickle", "wb"))

data = dataLoader("./Data/cnnkeystroke.pickle")
dataNN = dataLoader("./Data/dataBenchmarkTf.pickle")
print(dataNN.data.shape, dataNN.data_labels.shape)
print(data.data.shape, data.data_labels.shape)
print(data.train.shape, data.test.shape, data.valid.shape)
print(data.valid_classes, dataNN.valid_classes)

inp = tf.placeholder(tf.float32, shape=[None, 1, 31, 1])
out = tf.placeholder(tf.float32, shape = [None, 51])
inpNN = tf.placeholder(tf.float32, shape=[None, 31])

def getLayer(shape):
    w, b = tf.Variable(tf.truncated_normal([shape[0], shape[1]], stddev=0.1)), tf.Variable(tf.zeros([shape[1]]))
    return w, b

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([1, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(inp, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([1, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

print(h_pool2.shape)
W_fc1 = weight_variable([1 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 1*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 51])
b_fc2 = bias_variable([51])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#feed_forward
w, b = getLayer([31, 80])
w1, b1 = getLayer([80, 80])
w2, b2 = getLayer([80, 60])
w0, b0 = getLayer([60, 51])

h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(inpNN, w) + b), keep_prob)
h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w1) + b1), keep_prob)
h3 = tf.nn.dropout(tf.nn.relu(tf.matmul(h2, w2) + b2), keep_prob)
y_ff = tf.nn.relu(tf.matmul(h3, w0) + b0)

#final layers
final_inp = tf.concat([y_ff, y_conv], axis = 1)
print(final_inp.shape)

w1, b1 = getLayer([102, 200])
w11, b11 = getLayer([200, 200])
w21, b21 = getLayer([200, 100])
w01, b01 = getLayer([100, 51])

h11 = tf.nn.dropout(tf.nn.relu(tf.matmul(final_inp, w1) + b1), keep_prob)
h21 = tf.nn.dropout(tf.nn.relu(tf.matmul(h11, w11) + b11), keep_prob)
h31 = tf.nn.dropout(tf.nn.relu(tf.matmul(h21, w21) + b21), keep_prob)

y_final = tf.nn.relu(tf.matmul(h31, w01) + b01)

y_ = tf.nn.softmax(y_final)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=out, logits=y_final))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_final, 1), tf.argmax(out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batchSize = 128
    for epoch in range(500):
        j = -1
        i = 0
        while i < data.train.shape[0]:
            j += 1
            batch = data.train[i: i + batchSize]
            batchNN = dataNN.train[i: i + batchSize]
            batch_labels = data.train_labels[i: i + batchSize]
            i = i + batchSize
            #print(batch.shape)
            train_step.run(feed_dict={inp: batch, out: batch_labels, inpNN: batchNN, keep_prob: 0.5})
            if not j%50:
                train_accuracy = accuracy.eval(feed_dict={inp: batch, out: batch_labels, inpNN: batchNN, keep_prob: 1.0})
                train_out = sess.run(y_, feed_dict={inp: batch, out: batch_labels, inpNN: batchNN, keep_prob: 1.0})
                valid_accuracy = accuracy.eval(feed_dict={inp: data.valid, out: data.valid_labels, inpNN: dataNN.valid, keep_prob: 1.0})
                valid_out = sess.run(y_, feed_dict={inp: data.valid, out: data.valid_labels, inpNN: dataNN.valid, keep_prob: 1.0})
                #print(valid_out[0])
                test_accuracy = accuracy.eval(feed_dict={inp: data.test, out: data.test_labels, inpNN: dataNN.test, keep_prob: 1.0})
                test_out = sess.run(y_, feed_dict={inp: data.test, out: data.test_labels, inpNN: dataNN.test, keep_prob: 1.0})
                TEER = GetEER(test_out, dataNN.test_labels)
                VEER = GetEER(valid_out, dataNN.valid_labels)
                print(epoch, j, train_accuracy, valid_accuracy, test_accuracy, VEER, TEER)
