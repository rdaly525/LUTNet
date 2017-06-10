import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *

import matplotlib.pyplot as plt

if __name__ == '__main__':


  sigma = 1
  N = 4
  bits = 4
  data = datasets.adddata(N,bits)
  test_data = data.test_data
  lr = 0.1
  rw = 0.01
  layers = 7
  
  Xbits = 2*bits
  ybits = bits+1

  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
  
  Ws = []
  curl = X
  curbits = 2*bits
  for li in range(layers-1):
    nextbits = int(Xbits - (li*1.0/(Xbits+1-ybits)))
    print "C,N =", curbits, nextbits
    curl, Wsl = lutlayer(N,sigma,curbits,nextbits)(curl)
    Ws += Wsl
    curbits = nextbits
  print "C,N =",curbits,ybits
  y, Wfinal = lutlayer(N,sigma,curbits,ybits)(curl)
  Ws += Wfinal
  #l1, Ws1 = lutlayer(N,sigma,2*bits,2*bits)(X)
  #l2, Ws2 = lutlayer(N,sigma,2*bits,2*bits)(l1)
  #l3, Ws3 = lutlayer(N,sigma,2*bits,2*bits-1)(l2)
  #y, Ws4 = lutlayer(N,sigma,2*bits-1,bits+1)(l3)
  #Ws = Ws1 + Ws2 + Ws3 + Ws4
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

  sample = 50
  iters = 5000
  losses = np.zeros(iters/sample)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    wval = None
    for i in range(iters):
      tdata = data.next_data(16)
      _,yval,lossval = sess.run([train_step,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
      if (i%sample==0):
        print lossval
        print "  ",scaleto01(tdata[0][0][0:bits]),"+",scaleto01(tdata[0][0][bits:]),"=",scaleto01(tdata[1][0])
        print "  lrn",scaleto01(yval[0],False)
        losses[i/sample] = lossval
        #print "  ",scaleto01(tdata[0][0][0:bits])[::-1],"+",scaleto01(tdata[0][0][bits:])[::-1],"=",scaleto01(tdata[1][0])[::-1]
        #print "  lrn,cor",scaleto01(yval[0],False)[::-1]

    print "Accuracy!"
    print accuracy.eval(feed_dict={X:test_data[0],y_:test_data[1]})
  
  plt.figure(1)
  plt.plot(losses)
  plt.xlabel("iter/10")
  plt.ylabel("loss")
  plt.show()
