import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *


if __name__ == '__main__':


  def xor(x,N):
    ret = 0
    for i in range(N):
      ret = ret ^ ((x>>i)&1)
    return ret
  
  def andr(x,N):
    ret = 1
    for i in range(N):
      ret = ret & ((x>>i)&1)
    return ret

  def mod5(x,N):
    return int(x%5==0)


  sigma = 0.1
  N = 5
  data = datasets.lutdata(N,mod5)
  lr = 0.1
  rw = 0.01
  x = tf.placeholder(tf.float32, shape=[None,N])
  y_ = tf.placeholder(tf.float32, shape=[None])
  y, W = lutN(N,sigma)(x)
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

