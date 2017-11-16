import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *


if __name__ == '__main__':


  sigma = 1
  N = 4
  K = 13
  selval = 7
  data = datasets.Seldata(K,selval)
  test_data = data.test
  print(data.next_data(3))
  lr = 0.1
  rw = 0.01
  X = tf.placeholder(tf.float32, shape=[None,K])
  y_ = tf.placeholder(tf.float32, shape=[None])
  print(X)
  kind = "gaussian"
  #kind = "triangle"
  y, W = SelectK(K,kind)(X)
  print(y,y_,W)
  loss = tf.nn.l2_loss(y-y_) + rw*binary_reg(W)
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  
  yscale = y > 0
  y_scale = y_ > 0
  correct_pred = tf.equal(yscale,y_scale)
  accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
 
  sample = 20
  iters = 500
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    wval = None
    for i in range(iters):
      tdata = data.next_data(32)
      _,yval,lossval = sess.run([train_step,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
      if (i%sample==0):
        print(lossval, "("+str(i)+"/"+str(iters)+")")
        print("  ",scaleto01(tdata[0][0]),"=",scaleto01(tdata[1][0]))
        print("  lrn",yval[0])
        print("  acc",accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs}))
      if (i%(sample*4)==39):
        curWs = sess.run([W],feed_dict={X:test_data.inputs,y_:test_data.outputs})
        print("CURWs", curWs)
        yV,_yV = sess.run([yscale,y_scale],feed_dict={X:test_data.inputs,y_:test_data.outputs})
        print("yV",yV)
        print("_yv", _yV)
    print("Accuracy!")
    print(accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs}))
    curWs = sess.run([W],feed_dict={X:test_data.inputs,y_:test_data.outputs})
    print("CURWs", curWs)

