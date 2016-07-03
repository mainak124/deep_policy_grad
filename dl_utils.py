import tensorflow as tf

def _conv(X, nIn, nOut, ker, st, padType, bias_val=0.0):
    with tf.name_scope('conv') as scope, tf.device('/gpu:1'):
    #with tf.name_scope(name) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=1e-2), shape=[ker, ker, nIn, nOut], dtype=tf.float32, trainable=True, name='weights')
        conv = tf.nn.conv2d(X, kernel, [1, st, st, 1], padding=padType)
        biases = tf.get_variable(initializer=tf.constant_initializer(bias_val), shape=[nOut], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        return conv1

def _pool(X, ker, st, padType):
    pool_counter += 1
    with tf.name_scope('pool') as scope, tf.device('/gpu:1'):
    #with tf.name_scope(name) as scope:
        return tf.nn.max_pool(X, ksize=[1, ker, ker, 1], strides=[1, st, st, 1], padding=padType, name=scope)

def _norm(l_input, lsize=5, alpha=0.0001, beta=0.75):
    with tf.name_scope('lrn') as scope:
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=alpha, beta=beta, name=scope)        

def _fc(X, nIn, nOut, std, activation=True, bias_val = 0.0):
    with tf.name_scope('fc') as scope, tf.device('/gpu:1'):
    #with tf.name_scope(name) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=std), shape=[nIn, nOut], dtype=tf.float32, trainable=True, name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(bias_val), shape=[nOut], dtype=tf.float32, trainable=True, name='biases')
        if (activation==True):
            fc1 = tf.nn.relu_layer(X, kernel, biases, name=name)
        else:    
            fc1 = tf.nn.bias_add(tf.matmul(X, kernel), biases, name=name)
        return fc1

def _dropout(X, keep_prob, is_train):
    #with tf.name_scope(name) as scope, tf.device('/gpu:1'):
    with tf.name_scope('dropout') as scope:
        if is_train:
            return tf.nn.dropout(X, keep_prob)
        else:
            return X*keep_prob    

def _softmax(X, Y, nIn, nOut):
    global softmax_counter
    #global parameters
    name = 'softmax' + str(softmax_counter)
    softmax_counter += 1
    with tf.name_scope(name) as scope, tf.device('/gpu:1'):
    #with tf.name_scope(name) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=1e-2), shape=[nIn, nOut], trainable=True, dtype=tf.float32, name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[nOut], dtype=tf.float32, trainable=True, name='biases')
        softmax1 = tf.nn.softmax(tf.matmul(X, kernel) + biases, name=name)
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(X, kernel) + biases, Y, name=name))
        #cost = -tf.reduce_sum(Y*tf.log(softmax1))
        #parameters += [kernel, biases]
        params = [kernel, biases]
        return params, cost, softmax1

