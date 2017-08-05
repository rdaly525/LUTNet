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

  def fun(x,N):
    return int(math.sin(x*308.234324)*math.cos(x**2.3) + 1)

  sigma = 1
  N = 4
  data = datasets.Lutdata(N,xor)
  test_data = data.test
  lr = 0.1
  rw = 0.01
  X = tf.placeholder(tf.float32, shape=[None,N])
  y_ = tf.placeholder(tf.float32, shape=[None])
  W = Var(2**N)
  Wph = tf.placeholder(tf.float32,shape=[2**N])
  y, W = LutN(N)(X)
  W_assign = tf.assign(W,Wph)
  loss = tf.nn.l2_loss(y-y_) + rw*binary_reg(W)
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  
  yscale = y > 0
  y_scale = y_ > 0
  correct_pred = tf.equal(yscale,y_scale)
  accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
 
  sample = 1
  iters = 50
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    wval = None
    for i in range(iters):
      tdata = data.next_data(16)
      _,yval,lossval = sess.run([train_step,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
      if (i%sample==0):
        print(lossval, "("+str(i)+"/"+str(iters)+")")
        print("  ",scaleto01(tdata[0][0]),"=",scaleto01(tdata[1][0]))
        print("  lrn",yval[0])
        print("  acc",accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs}))
      if (i%(sample*1)==0):
        curWs = sess.run([W],feed_dict={X:test_data.inputs,y_:test_data.outputs})
        print("CURWs", curWs)
        #yV,_yV = sess.run([yscale,y_scale],feed_dict={X:test_data.inputs,y_:test_data.outputs})
        #print("yV",yV)
        #print("_yv", _yV)
    print("Accuracy!")
    print(accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs}))
    curWs = sess.run([W])
    print("CUR",curWs)
    fd = make_feed_dict([Wph],[curWs])
    print("FD",fd)
    print(sess.run([W_assign],feed_dict=fd))
    print(sess.run([W]))
    print(accuracy.eval(feed_dict={X:test_data.inputs,y_:test_data.outputs}))

