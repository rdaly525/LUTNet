import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

  image_width = 12

  data = datasets.Mnistdata(image_width=image_width)
  sigma = 1
  N = 4
  lr = 0.1
  rw = 0.01
  Xbits = image_width**2
  layers = [Xbits,100,50,25,15,10]
  ybits = 10
  

  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
  
  y, Ws = MacroLutLayer(N,layers)(X)
  
  totLuts = 0
  for w in Ws:
    shape = w.get_shape().as_list()
    totLuts += shape[0]*shape[1]
  print("Total Luts", totLuts//2**N)

  loss = tf.nn.l2_loss(y-y_) + rw*binary_reg(Ws)
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  train_step2 = tf.train.GradientDescentOptimizer(lr//10).minimize(loss)
  
  yscale = y > 0
  y_scale = y_ > 0
  correct_pred = tf.reduce_all(tf.equal(yscale,y_scale),1)
  accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

  sample = 20
  iters = 4000
  batch = 32
  losses = np.zeros(iters//sample)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    wval = None
    for i in range(iters):
      tdata = data.next_data(batch)
      yval,lossval = None,None
      if (i < 2000):
        _,yval,lossval = sess.run([train_step,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
      else:
        _,yval,lossval = sess.run([train_step2,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
      if (i%sample==0):
        print(lossval, "("+str(i)+"/"+str(iters)+")")
        print("  cor",scaleto01(tdata[1][0]))
        print("  lrn",scaleto01(yval[0],False))
        losses[i//sample] = lossval
    print("Accuracy!")
    print(accuracy.eval(feed_dict={X:data.test[0],y_:data.test[1]}))

  plt.figure(1)
  plt.plot(losses)
  plt.xlabel("iter/"+str(sample))
  plt.ylabel("loss")
  plt.show()
