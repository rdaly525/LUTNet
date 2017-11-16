from __future__ import division
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
  kind = "gaussian"
  kind = "triangle"
  
  #Plus triangle
  #sigma = 1
  #N = 4
  #lr = 0.1
  #rw = 0.01
  #layers = 6
  
  bits = 3
  Xbits = 2*bits
  ybits = bits+1
  
  def add(a,b):
    return a+b
  
  #def popcnt(a,b):
  #  cnt = 0
  #  for i in range(bits):
  #    cnt += ((a>>i)&1) + ((b>>i)&1)
  #  return cnt

  data = datasets.Binopdata(add,Xbits//2,ybits)
  test_data = data.test
  print(test_data.inputs[6], test_data.outputs[6])
  data.next_data(6)
  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
  
  c0 = tf.expand_dims(tf.fill(tf.shape(X[:,0]),-1.0),1)
  l0, selWs0, lutWs0 = SelectLutLayer(Xbits,N,4,kind,sigma=1)(X)
  l1, selWs1, lutWs1 = SelectLutLayer(4+Xbits,N,4,kind,sigma=1)(tf.concat([X,l0],axis=1))
  y, selWs2 = Selects(8,4,kind,sigma=1)(tf.concat([l0,l1],axis=1))
  
  
  selWs = flatten_list(selWs0)+flatten_list(selWs1) + selWs2
  lutWs = lutWs0 + lutWs1
  Ws = selWs + lutWs 
  Wphs = [tf.placeholder(tf.float32,shape=W.get_shape()) for W in Ws]
  W_assigns = [tf.assign(Ws[i],Wphs[i]) for i in range(len(Ws))] 
  loss = tf.nn.l2_loss(y-y_) + rw*(
    2*binary_reg(flatten_list(selWs0)) +
    2*binary_reg(flatten_list(selWs1)) +
    2*binary_reg(selWs2) +
    binary_reg(lutWs0) +
    binary_reg(lutWs1)
  )
  
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  train_step_2 = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=lutWs)
  #train_step = tf.train.AdamOptimizer(lr).minimize(loss)
  
  yscale = y > 0
  y_scale = y_ > 0
  correct_pred = tf.equal(yscale,y_scale)
  correct_red = tf.reduce_all(correct_pred,axis=1)
  accuracy = tf.reduce_mean(tf.cast(correct_red,tf.float32))
  sample = 20
  iters = 200
  qit = 10
  losses = np.zeros(iters*qit//sample)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    wval = None
    for q in range(qit):
      if (q < 5):
        print("Quantizing", q)
        curWs = sess.run(Ws)
        fd = make_feed_dict(Wphs,curWs,0.9)
        sess.run(W_assigns,feed_dict=fd)
      for i in range(iters):
        tdata = data.next_data(32)
        if (q < 5):
          _,yval,lossval = sess.run([train_step,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
        else:
          _,yval,lossval = sess.run([train_step_2,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
        if ((iters*q+i)%sample==0):
          print(lossval, "("+str(i)+"/"+str(iters)+")")
          print("  ",scaleto01(tdata[0][0][0:bits]),"+",scaleto01(tdata[0][0][bits:]),"=",scaleto01(tdata[1][0]))
          print("  lrn",scaleto01(yval[0],False))
          print("  ac", accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs}))
          losses[(iters*q+i)//sample] = lossval
      #q = q/3
    print("Quantizing Last")
    curWs = sess.run(Ws)
    fd = make_feed_dict(Wphs,curWs,0,1)
    sess.run(W_assigns,feed_dict=fd)

    print("Accuracy!")
    print(accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs}))
    print("sw0",sess.run(flatten_list(selWs0)))
    print("sw1",sess.run(flatten_list(selWs1)))
    print("lw0",sess.run(lutWs0))
    print("lw1",sess.run(lutWs1))
  
  plt.figure(1)
  plt.plot(losses)
  plt.xlabel("iter/"+str(sample))
  plt.ylabel("loss")
  plt.show()
