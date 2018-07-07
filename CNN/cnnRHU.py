import pickle
data = pickle.load(open("../Data/RHUcleaned.pickle", "rb"))
#{"data": datax, "labels": yOneHot, "classes": datay}
import tensorflow as tf
from CalculateEER import SKGetEER as GetEER
from sklearn.model_selection import train_test_split
import numpy as np

X = np.array(data['data'])
Y = np.array(data['labels'])
numFeatures = X.shape[1]
numLabels = max(data['classes']) + 1
X = X.reshape(X.shape[0], 1, numFeatures, 1)

n = X.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
print(X_train.shape)

tf.reset_default_graph()
inp = tf.placeholder(tf.float32, shape=[None, 1, numFeatures, 1])
out = tf.placeholder(tf.float32, shape = [None, numLabels])


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

W_conv1 = weight_variable([1, 5, 1, numFeatures + 1])
b_conv1 = bias_variable([numFeatures + 1])

h_conv1 = tf.nn.relu(conv2d(inp, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([1, 5, numFeatures + 1, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

print(h_pool2.shape)
W_fc1 = weight_variable([1 * 14 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 1*14*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, numLabels])
b_fc2 = bias_variable([numLabels])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_ = tf.nn.softmax(y_conv)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=out, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
try:
    sess.run(tf.global_variables_initializer())
    batchSize = 128
    for epoch in range(500):
        j = -1
        i = 0
        while i < X_train.shape[0]:
            j += 1
            batch = X_train[i: i + batchSize]
            batch_labels = y_train[i: i + batchSize]
            i = i + batchSize
            #print(batch.shape)
            train_step.run(feed_dict={inp: batch, out: batch_labels, keep_prob: 0.5}, session = sess)
            if not j%50:
                train_accuracy = accuracy.eval(feed_dict={inp: batch, out: batch_labels, keep_prob: 1.0}, session = sess)
                train_out = sess.run(y_, feed_dict={inp: batch, out: batch_labels, keep_prob: 1.0})
                valid_accuracy = accuracy.eval(feed_dict={inp: X_valid, out: y_valid, keep_prob: 1.0}, session = sess)
                valid_out = sess.run(y_, feed_dict={inp: X_valid, out: y_valid, keep_prob: 1.0})
                #print(valid_out[0])
                test_accuracy = accuracy.eval(feed_dict={inp: X_test, out: y_test, keep_prob: 1.0}, session = sess)
                test_out = sess.run(y_, feed_dict={inp: X_test, out: y_test, keep_prob: 1.0})
                TEER = GetEER(test_out, y_test)
                VEER = GetEER(valid_out, y_valid)
                print(epoch, j, "ta", train_accuracy, "va", valid_accuracy, "tea", test_accuracy, "eer:", VEER, TEER)
except Exception as e:
    print(e)
    d = {}
