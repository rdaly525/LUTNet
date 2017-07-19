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
  
  bits = 7
  Xbits = bits
  ybits = 3
  
  def popcnt(a):
    cnt = 0
    for i in range(bits):
      cnt += ((a>>i)&1)
    return cnt

  data = datasets.Unaryopdata(popcnt,Xbits,ybits)
  test_data = data.test
  print test_data.inputs[6], test_data.outputs[6]
  data.next_data(6)
  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
  c0 = tf.fill(tf.shape(X[:,0]),-1.0)

  lut4 = LutN(4,kind="triangle")
  Ws = [None for i in range(9)]
  l00,Ws[0] = lut4([X[:,0],X[:,2],X[:,4],c0])
  l01,Ws[1] = lut4([X[:,0],X[:,2],X[:,4],c0])
  l02,Ws[2] = lut4([X[:,1],X[:,3],X[:,5],c0])
  l03,Ws[3] = lut4([X[:,1],X[:,3],X[:,5],c0])

  l10,Ws[4] = lut4([l00,l01,l02,X[:,6]])
  l11,Ws[5] = lut4([X[:,2],l02,X[:,4],X[:,6]])
  
  l20,Ws[6] = lut4([X[:,0],l11,c0,c0])
  l21,Ws[7] = lut4([l01,l10,l02,l03])
  l22,Ws[8] = lut4([l10,l03,c0,c0])
  
  y = tf.stack([l20,l22,l21],axis=1)
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
      tdata = data.next_data(31,False)
      _,yval,lossval = sess.run([train_step,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
      if (i%sample==0):
        print lossval, "("+str(i)+"/"+str(iters)+")"
        print "  ",scaleto01(tdata[0][0]),"=",scaleto01(tdata[1][0])
        print "  lrn",scaleto01(yval[0],False)
        losses[i/sample] = lossval
        print "  acc",accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs})

    print "Accuracy!"
    print accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs})
    #yv, yv_, cor = sess.run([y,y_,correct_pred],feed_dict={X:test_data.inputs,y_:test_data.outputs})
    #print "cor",cor
    #print "yv",yv
    #print "yv_",yv_
  plt.figure(1)
  plt.plot(losses)
  plt.xlabel("iter/"+str(sample))
  plt.ylabel("loss")
  plt.show()
