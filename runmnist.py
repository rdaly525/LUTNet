import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

  data = datasets.Mnistdata(ds=2)
  print "FUCK"
  sigma = 1
  N = 4
  lr = 0.1
  rw = 0.01
  layers = 10
  Xbits = 14*14
  ybits = 10

  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
  
  y, Ws = lutlayers(N,sigma,Xbits,ybits,layers)(X)
  
  print "Total luts", len(Ws)
  loss = tf.nn.l2_loss(y-y_) + rw*binary_reg(Ws)
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  
  yscale = y > 0
  y_scale = y_ > 0
  correct_pred = tf.reduce_all(tf.equal(yscale,y_scale),1)
  accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

  sample = 20
  iters = 5000
  losses = np.zeros(iters/sample)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    wval = None
    for i in range(iters):
      tdata = data.next_data(32)
      _,yval,lossval,co_ped,ac= sess.run([train_step,y,loss,correct_pred,accuracy],feed_dict={X:tdata[0],y_:tdata[1]})
      if (i%sample==0):
        print lossval, "("+str(i)+"/"+str(iters)+")"
        print "  cor",scaleto01(tdata[1][0])
        print "  lrn",scaleto01(yval[0],False)
        losses[i/sample] = lossval
        print "co,ac",co_ped,ac
    print "Accuracy!"
    print accuracy.eval(feed_dict={X:data.test[0],y_:data.test[1]})
    print sess.run(Ws[0])
  plt.figure(1)
  plt.plot(losses)
  plt.xlabel("iter/"+str(sample))
  plt.ylabel("loss")
  plt.show()
