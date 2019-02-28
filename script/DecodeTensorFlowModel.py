import tensorflow as tf
import NeuralNetworkSaver as nns

nx = 94
n1 = 256
n2 = 64
# n3 = 32
n4 = 11

with tf.variable_scope("Layer1"):
    w1 = tf.Variable(tf.random_normal([nx, n1]), name="weight_1")
    b1 = tf.Variable(tf.random_normal([1, n1]), name="bias_1")
    # o1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1, name="o1"))

with tf.variable_scope("Layer2"):
    w2 = tf.Variable(tf.random_normal([n1, n2]), name="weight_2")
    b2 = tf.Variable(tf.random_normal([1, n2]), name="bias_2")
    # o2 = tf.nn.relu(tf.add(tf.matmul(o1, w2), b2, name="o2"))

with tf.variable_scope("Out"):
    w4 = tf.Variable(tf.random_normal([n2, n4]), name="weight_out")
    b4 = tf.Variable(tf.random_normal([1, n4]), name="bias_out")
    # o4 = (tf.add(tf.matmul(o2, w4), b4, name="o4"))

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './logs/SaveforLoad/SL.ckpt')

    _w1, _w2, _w4, _b1, _b2, _b4 = sess.run([w1, w2, w4, b1, b2, b4])
    nns.NeuralNetworkSaver([nx, n1, n2, n4], [["relu", _w1, _b1], ["relu", _w2, _b2], ["linear", _w4, _b4]], "test")
