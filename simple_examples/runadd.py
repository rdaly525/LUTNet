import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *

import matplotlib.pyplot as plt

if __name__ == '__main__':


  sigma = 1
  N = 4
  lr = 0.1
  rw = 0.01
  layers = 6
  
  bits = 3
  Xbits = 2*bits
  ybits = 6
  
  def add(a,b):
    return a*b
  
  def popcnt(a,b):
    cnt = 0
    for i in range(bits):
      cnt += ((a>>i)&1) + ((b>>i)&1)
    return cnt

  data = datasets.Binopdata(add,Xbits/2,ybits)
  test_data = data.test
  print test_data.inputs[6], test_data.outputs[6]
  data.next_data(6)
  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
 
  y, Ws = lutlayers(N,sigma,Xbits,ybits,layers)(X)

  print "Total Luts =", len(Ws)
  loss = tf.nn.l2_loss(y-y_) + rw*binary_reg(Ws)
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  
  yscale = y > 0
  y_scale = y_ > 0
  correct_pred = tf.equal(yscale,y_scale)
  accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


  sample = 20
  iters = 2000
  losses = np.zeros(iters/sample)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    wval = None
    for i in range(iters):
      tdata = data.next_data(32)
      _,yval,lossval = sess.run([train_step,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
      if (i%sample==0):
        print lossval, "("+str(i)+"/"+str(iters)+")"
        print "  ",scaleto01(tdata[0][0][0:bits]),"+",scaleto01(tdata[0][0][bits:]),"=",scaleto01(tdata[1][0])
        print "  lrn",scaleto01(yval[0],False)
        losses[i/sample] = lossval

    print "Accuracy!"
    print accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs})
    print sess.run(Ws[0])
  plt.figure(1)
  plt.plot(losses)
  plt.xlabel("iter/"+str(sample))
  plt.ylabel("loss")
  plt.show()
