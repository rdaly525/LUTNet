import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

  image_width = 28

  data = datasets.Mnistdata(image_width=image_width)
  print (data.test(False)[1][5])
  sigma = 1
  N = 4
  lr = 0.003
  rw = 0.0001
  Xbits = image_width**2
  layers = [Xbits,600,400,250,150,100,60]
  #layers = [Xbits,150,100,75,50,30,30,20,20,15,15,10]
  ybits = 10
  

  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
  
  y, Ws = MacroLutLayer(N,layers)(X)
  print (y)
  y = tf.reshape(y,[-1,10,6])
  print (y)
  y = tf.reduce_sum(y,2)
  print (y)

  totLuts = 0
  print (len(Ws))
  for w in Ws:
    print(w)
    shape = w.get_shape().as_list()
    totLuts += shape[0]*shape[1]
  print("Total Luts", totLuts//2**N)

  Wphs = [tf.placeholder(tf.float32,shape=w.shape) for w in Ws]
  W_assigns = [tf.assign(Ws[i],Wphs[i]) for i in range(len(Ws))]


  #sm = tf.nn.softmax(y)
  #print("sm",sm)
  #loss = tf.nn.l2_loss(y-y_) + rw*binary_reg(Ws)
  loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
  #print(loss1)
  loss = loss1 + rw*binary_reg(Ws)
  train_step = tf.train.AdamOptimizer(lr).minimize(loss)
  train_step2 = tf.train.AdamOptimizer(lr//10).minimize(loss)

  ymax = tf.reduce_max(y,axis=1)
  ymax = tf.reshape(ymax,[-1,1])
  ymax = tf.tile(ymax,[1,10])
  yscale = y >= ymax
  y_scale = tf.cast(y_,tf.bool)
  print("1",ymax)
  print("2",y)
  print("3",yscale)
  print("4",y_scale)
  
  correct_pred = tf.cast(tf.reduce_all(tf.equal(yscale,y_scale),1),tf.float32)
  accuracy = tf.reduce_mean(correct_pred)
  #


  sample = 20
  iters = 1000
  batch = 32
  qiter = 6
  losses = np.zeros((iters*qiter)//sample)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for j in range(qiter):
      for i in range(iters):
        tdata = data.next_data(batch)
        yval,lossval = None,None
        y1,y2 = None,None
        #if (j < qiter//2):
          #_,yval,lossval,sm_val = sess.run([train_step,y,loss,sm],feed_dict={X:tdata[0],y_:tdata[1]})
        tdata1 = scaleto01(tdata[1])
          #print(tdata1)
        _,yval,lossval,y1,y2,acc = sess.run([train_step,y,loss,yscale, y_scale,accuracy],feed_dict={X:tdata[0],y_:tdata1})
          #print("1",y1)
          #print("2",y2)
          #print("3",y3)
          #print("4",y4)
        #else:
        #  _,yval,lossval = sess.run([train_step,y,loss],feed_dict={X:tdata[0],y_:tdata[1]})
        if (i%sample==0):
          print(lossval, j, "("+str(i)+"/"+str(iters)+")")
          print("  cor",scaleto01(tdata[1][0]))
          print("  lrn",(yval[0]*.5))
          print("  ysca",y1[0])
          print("  y_sca",y2[0])
          print("  acc",acc)
          #print("ys",ys_val)
          #print("y_s",y_s_val)
          #print("pred",pred_val)
          #print("ac",ac_val)
          losses[(i+j*iters)//sample] = lossval
      print(j, "Accuracy_test")
      print(accuracy.eval(feed_dict={X:data.test(False)[0],y_:data.test(False)[1]}))
      #print(j, "Accuracy_train")
      #print(accuracy.eval(feed_dict={X:data.train(False)[0],y_:data.train(False)[1]}))

      curWs = sess.run(Ws,feed_dict={X:data.test(False)[0],y_:data.test(False)[1]})
      print("curW5",curWs[2])
      if (j==qiter-1):
        fd = make_feed_dict(Wphs,curWs)
      else:
        fd = make_feed_dict(Wphs,curWs,0.9)
      #print("FD",fd)
      sess.run(W_assigns,feed_dict=fd)
      tdata = data.next_data(batch)
      curWs = sess.run(Ws,feed_dict={X:tdata[0],y_:tdata[1]})
      print("newW5", curWs[2])

      print(j, "Accuracy_test_q")
      print(accuracy.eval(feed_dict={X:data.test(False)[0],y_:data.test(False)[1]}))
      #print(j, "Accuracy_train_q")
      #print(accuracy.eval(feed_dict={X:data.train(False)[0],y_:data.train(False)[1]}))

  plt.figure(1)
  plt.plot(losses)
  plt.xlabel("iter/"+str(sample))
  plt.ylabel("loss")
  plt.show()
