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
  layers = 7
  
  bits = 8
  Xbits = 2*bits
  ybits = 5
  
  def add(a,b):
    return (a+b)%(2**bits)
  
  def popcnt(a,b):
    cnt = 0
    for i in range(bits):
      cnt += ((a>>i)&1) + ((b>>i)&1)
    return cnt

  data = datasets.binopdata(N,popcnt,Xbits/2,ybits)
  test_data = data.test_data
  data.next_data(6)
  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
  
  Ws = []
  curl = X
  curbits = Xbits
  for li in range(layers-1):
    nextbits = int(Xbits - (li*1.0*(Xbits+1-ybits))/(layers))
    curl, Wsl = lutlayer(N,sigma,curbits,nextbits)(curl)
    Ws += Wsl
    curbits = nextbits
  y, Wfinal = lutlayer(N,sigma,curbits,ybits)(curl)
  Ws += Wfinal
  
  print "Total Luts =", len(Ws)
  loss = tf.nn.l2_loss(y-y_) + rw*binary_reg(Ws)
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  
  yscale = y > 0
  y_scale = y_ > 0
  correct_pred = tf.equal(yscale,y_scale)
  accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

  def descW(W):
    dw = (W > 0).astype(int)
    return dw

  sample = 20
  iters = 3000
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
    print accuracy.eval(feed_dict={X:test_data[0],y_:test_data[1]})
    print sess.run(Ws[0])
  plt.figure(1)
  plt.plot(losses)
  plt.xlabel("iter/"+str(sample))
  plt.ylabel("loss")
  plt.show()
