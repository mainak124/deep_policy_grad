import tensorflow as tf
import config

global device
device = '/gpu:0'

def _conv(X, nIn, nOut, ker, st, padType, bias_val=0.0):
    with tf.name_scope('conv') as scope, tf.device(device):
    #with tf.name_scope(name) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=1e-2), shape=[ker, ker, nIn, nOut], dtype=tf.float32, trainable=True, name='weights')
        conv = tf.nn.conv2d(X, kernel, [1, st, st, 1], padding=padType)
        biases = tf.get_variable(initializer=tf.constant_initializer(bias_val), shape=[nOut], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        return conv1

def _pool(X, ker, st, padType):
    pool_counter += 1
    with tf.name_scope('pool') as scope, tf.device(device):
    #with tf.name_scope(name) as scope:
        return tf.nn.max_pool(X, ksize=[1, ker, ker, 1], strides=[1, st, st, 1], padding=padType, name=scope)

def _norm(l_input, lsize=5, alpha=0.0001, beta=0.75):
    with tf.name_scope('lrn') as scope:
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=alpha, beta=beta, name=scope)        

def _fc(X, nIn, nOut, std, activation=True, bias_val = 0.0):
    with tf.name_scope('fc') as scope, tf.device(device):
    #with tf.name_scope(name) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=std), shape=[nIn, nOut], dtype=tf.float32, trainable=True, name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(bias_val), shape=[nOut], dtype=tf.float32, trainable=True, name='biases')
        if (activation==True):
            fc1 = tf.nn.relu_layer(X, kernel, biases, name='fc')
        else:    
            fc1 = tf.nn.bias_add(tf.matmul(X, kernel), biases, name='fc')
        return fc1

def _dropout(X, keep_prob, is_train):
    #with tf.name_scope(name) as scope, tf.device('/gpu:1'):
    with tf.name_scope('dropout') as scope:
        if is_train:
            return tf.nn.dropout(X, keep_prob)
        else:
            return X

def _softmax(X, Y, nIn, nOut):
    with tf.name_scope('sm') as scope, tf.device(device):
    #with tf.name_scope(name) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=1e-2), shape=[nIn, nOut], trainable=True, dtype=tf.float32, name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[nOut], dtype=tf.float32, trainable=True, name='biases')
        softmax1 = tf.nn.softmax(tf.matmul(X, kernel) + biases, name=name)
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(X, kernel) + biases, Y, name=name))
        return cost, softmax1
