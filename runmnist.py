import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

  hyp = dict(
    image_width = 20,
    learning_rate = 0.003,
    reg_weight = 0.00001,
    output_bits = 8,
    layer_count = 12,
  )

  output_bits = hyp['output_bits']

  data = datasets.Mnistdata(image_width=hyp['image_width'])
  print (data.train(False)[0].shape)
  print (data.test(False)[1][5])
   
  Xbits = hyp['image_width']**2
 
  lut_bits = 4
  ybits = 10
  layers = create_layers(Xbits,ybits*output_bits,hyp['layer_count'])
  print (layers)
  
  qiter = 20

  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
  
  y, Ws = MacroLutLayer(lut_bits,layers)(X)
  print (y)
  scale = np.ones([1,1,output_bits])
  #scale[0][0] = np.array([1,1,1,1,1,1])
  y = tf.reshape(y,[-1,10,output_bits]) #* scale
  print (y)
  y = tf.reduce_sum(y,2)
  print (y)

  totLuts = 0
  print (len(Ws))
  for w in Ws:
    print(w)
    shape = w.get_shape().as_list()
    totLuts += shape[0]*shape[1]
  print("Total Luts", totLuts//2**lut_bits)

  Wphs = [tf.placeholder(tf.float32,shape=w.shape) for w in Ws]
  W_assigns = [tf.assign(Ws[i],Wphs[i]) for i in range(len(Ws))]

  loss_pre = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
  losses = [loss_pre + hyp['reg_weight']*(100**(i/qiter))*binary_reg(Ws) for i in range(qiter)]

  train_steps = [tf.train.AdamOptimizer(hyp['learning_rate']).minimize(losses[i]) for i in range(qiter)]
  #train_steps = [tf.train.MomentumOptimizer(learning_rate,.5).minimize(losses[i]) for i in range(qiter)]
  #loss1 = loss_pre + 3*hyp['reg_weight']*binary_reg(Ws)
  #train_step = tf.train.AdamOptimizer(hyp['learning_rate']).minimize(loss)
  #train_step1 = tf.train.AdamOptimizer(hyp['learning_rate']).minimize(loss1)
  #train_step2 = tf.train.AdamOptimizer(hyp['learning_rate']//10).minimize(loss)

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

  sample = 20
  iters = 1720 #About 1 epoch
  batch = 32
  losslog = np.zeros((iters*qiter)//sample)
  hist = None
  nq_accuracy = np.zeros(qiter)
  uq_accuracy = np.zeros(qiter)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for j in range(qiter):
      for i in range(iters):
        tdata = data.next_data(batch)
        tdata1 = scaleto01(tdata[1])
        yval,lossval = None,None
        y1,y2 = None,None
        #if (j < qiter//2):
          #print(tdata1)
        _,yval,lossval,y1,y2,acc = sess.run([train_steps[j],y,losses[j],yscale, y_scale,accuracy],feed_dict={X:tdata[0],y_:tdata1})
          #print("1",y1)
          #print("2",y2)
          #print("3",y3)
          #print("4",y4)
        #else:
          #_,yval,lossval,y1,y2,acc = sess.run([train_step1,y,loss1,yscale, y_scale,accuracy],feed_dict={X:tdata[0],y_:tdata1})
        if (i%sample==0):
          print(lossval, j, "("+str(i)+"/"+str(iters)+")")
          print("  cor",scaleto01(tdata[1][0]))
          print("  lrn",(yval[0]))
          print("  ysca",y1[0])
          print("  y_sca",y2[0])
          print("  acc",acc)
          #print("ys",ys_val)
          #print("y_s",y_s_val)
          #print("pred",pred_val)
          #print("ac",ac_val)
          losslog[(i+j*iters)//sample] = lossval
      print(j, "Accuracy_test")
      uq_accuracy[j] = accuracy.eval(feed_dict={X:data.test(False)[0],y_:data.test(False)[1]})
      print(uq_accuracy[j])
      #print(j, "Accuracy_train")
      #print(accuracy.eval(feed_dict={X:data.train(False)[0],y_:data.train(False)[1]}))

      curWs = sess.run(Ws,feed_dict={X:data.test(False)[0],y_:data.test(False)[1]})
      hist = histogram(curWs)
      #plt.figure()
      #plt.hist(hist,bins=100)
      #plt.show() 
      print("curW5",curWs[2])
      fd = make_feed_dict(Wphs,curWs,1.0 - j/(qiter-1),True)
      #print("FD",fd)
      sess.run(W_assigns,feed_dict=fd)
      tdata = data.next_data(batch)
      curWs = sess.run(Ws,feed_dict={X:tdata[0],y_:tdata[1]})
      print("newW5", curWs[2])
      
      #hist = histogram(curWs)
      #plt.figure()
      #plt.hist(hist,bins=100)
      #plt.show() 

      print(j, "Accuracy_test_q")
      q_accuracy[j] = accuracy.eval(feed_dict={X:data.test(False)[0],y_:data.test(False)[1]})
      print(q_accuracy)
      #print(j, "Accuracy_train_q")
      #print(accuracy.eval(feed_dict={X:data.train(False)[0],y_:data.train(False)[1]}))

  plt.figure(1)
  plt.plot(losslog)
  plt.xlabel("iter/"+str(sample))
  plt.ylabel("loss")
  plt.show()
  plt.figure(2)
  plt.hist(hist,bins=100)
  plt.show()
  plt.figure(3)
  plt.plot(q_accuracy)
  plt.plot(uq_accuracy)
  plt.legend(['unquantized', 'quantized'], loc='upper left')
  plt.xlabel("qiter")
  plt.ylabel("accuracy")
