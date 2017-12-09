import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *
import matplotlib.pyplot as plt


def run_mnist():
  data = datasets.Mnistdata(image_width=28)
  print (data.train(False)[0].shape)
  print (data.test(False)[1][5])
   
  Xbits = 28**2
  ybits = 10

  x = tf.placeholder(tf.float32, shape=[None,Xbits])
  W = tf.Variable(tf.zeros([Xbits, ybits]))
  b = tf.Variable(tf.zeros([ybits]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  y_ = tf.placeholder(tf.float32, [None, ybits])
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  tdata = data.next_data(1)
  temp = []
  for i in range(28):
    temp.append(tdata[0][0][i*28:(i+1)*28])
  print(temp)
  print(scaleto01(tdata[1]))

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # Train
    for i in range(1000):
      tdata = data.next_data(100)
      tdata1 = scaleto01(tdata[1])
      sess.run(train_step,feed_dict={x:tdata[0],y_:tdata1})
      if i % 100 == 99:
        print(str(i) + ": " + str(sess.run(accuracy, feed_dict={x:data.test(False)[0],y_:data.test(False)[1]})))


    # Test trained model
    result = sess.run(accuracy, feed_dict={x:data.test(False)[0],y_:data.test(False)[1]})
    print(result)
  return result



if __name__ == '__main__':
  a = run_mnist()
  print(a)

  