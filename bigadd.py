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
  rw = 0.1
  
  bits = 6
  Xbits = 2*bits
  ybits = 7
  
  def add(a,b):
    return a+b

  data = datasets.Binopdata(add,Xbits/2,ybits)
  test_data = data.test
  print test_data.inputs[6], test_data.outputs[6]
  data.next_data(6)
  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
  c0 = tf.fill(tf.shape(X[:,0]),-1.0)
  kind="gaussian"
  #kind="triangle"
  lut4 = LutN(4,1,kind)
  Ws = [None for i in range(13)]
  
  l00,Ws[0] = lut4([X[:,0],X[:,3],X[:,6],c0])
  l01,Ws[1] = lut4([X[:,0],X[:,1],X[:,6],X[:,7]])
  l02,Ws[2] = lut4([X[:,1],X[:,7],c0,c0])

  l10,Ws[3] = lut4([X[:,0],l00,X[:,6],c0])
  l11,Ws[4] = lut4([X[:,2],l01,l02,X[:,8]])
  l12,Ws[5] = lut4([l01,X[:,2],X[:,3],X[:,8]])
  
  l20,Ws[6] = lut4([X[:,1],l10,X[:,7],c0])
  l21,Ws[7] = lut4([X[:,3],l12,X[:,9],c0])
  l22,Ws[8] = lut4([l12,X[:,2],X[:,3],X[:,9]])
  
  l30,Ws[9] = lut4([X[:,4],l22,X[:,10],c0])
  l31,Ws[10] = lut4([l22,X[:,4],X[:,10],c0])
  
  l40,Ws[11] = lut4([X[:,5],l31,X[:,11],c0])
  l41,Ws[12] = lut4([l31,l02,X[:,5],X[:,11]])
  
  Wphs = [tf.placeholder(tf.float32,shape=[2**N]) for i in range(13)]
  W_assigns = [tf.assign(Ws[i],Wphs[i]) for i in range(13)]

  print type(Ws[0]), "HERE"
  y = tf.stack([l00,l20,l11,l21,l30,l40,l41],axis=1)
  print "Total Luts =", len(Ws)
  loss = tf.nn.l2_loss(y-y_) + rw*binary_reg(Ws)
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  
  yscale = y > 0
  y_scale = y_ > 0
  correct_pred = tf.equal(yscale,y_scale)
  correct_red = tf.reduce_all(correct_pred,axis=1)
  accuracy = tf.reduce_mean(tf.cast(correct_red,tf.float32))


  sample = 20
  iters = 300
  losses = np.zeros(iters/sample)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    wval = None
    for i in range(iters):
      tdata = data.next_data(32)
      _,yval,lossval = sess.run([train_step,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
      if (i%sample==0):
        print lossval, "("+str(i)+"/"+str(iters)+")"
        #print "  ",scaleto01(tdata[0][0]),"=",scaleto01(tdata[1][0])
        #print "  lrn",yval[0]
        losses[i/sample] = lossval
        print "  acc",accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs})
      if (i%(sample*4)==39):
        curWs = sess.run(Ws,feed_dict={X:test_data.inputs,y_:test_data.outputs})
        print "CURWs", curWs
    print "Accuracy!"
    print accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs})
    #yv, yv_, cor = sess.run([y,y_,correct_pred],feed_dict={X:test_data.inputs,y_:test_data.outputs})
    
    curWs = sess.run(Ws,feed_dict={X:test_data.inputs,y_:test_data.outputs})
    fd = make_feed_dict(Wphs,curWs)
    print "FD",fd
    sess.run(W_assigns,feed_dict=fd)
    curWs = sess.run(Ws,feed_dict={X:test_data.inputs,y_:test_data.outputs})
    print "newWs", curWs
    print "Accuracy2!"
    print accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs})
    tdata = data.next_data(2)
    yval,lossval = sess.run([y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
    print lossval, yval
  
  plt.figure(1)
  plt.plot(losses)
  plt.xlabel("iter/"+str(sample))
  plt.ylabel("loss")
  #plt.show()
