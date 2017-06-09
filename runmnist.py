import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':

  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  
  H = 5
  W = 3
  Cin = 4
  Cout = 4
  
  N = 4
  lr = 0.1
  rw = 0.01
  x = tf.placeholder(tf.float32, shape=[None,H,W,Cin])
  l1,Ws = lutlayer(N,H,W,Cin,Cout,"depth")(x)
  print l1
  print len(Ws)
  assert(0)
  y_ = tf.placeholder(tf.float32, shape=[None,10])
  y, W = lutN(N,1)(x)
  loss = tf.nn.l2_loss(y-y_) + rw*binary_reg(W)
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  
  def descW(W):
    dw = (W > 0).astype(int)
    return dw

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter('logs',sess.graph)
    wval = None
    for i in range(1000):
      tdata = data.next_data(4)
      _,wval,yval,lossval = sess.run([train_step,W,y,loss],feed_dict={x:tdata[0],y_:tdata[1]})
      print lossval, yval,tdata[1]
      #print "  ",wval,tdata[0]
    print wval
    print "lrnd", descW(wval)
    print "corr", data.correct
    writer.close()

