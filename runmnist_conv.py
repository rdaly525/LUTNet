import numpy as np
import tensorflow as tf

from common import *
import datasets
from layers import *
import importlib
if importlib.find_loader('matplotlib.pyplot'):
  import matplotlib.pyplot as plt


def run_mnist_conv(hyp,display_graphs = False):
  assert hyp['quant_scheme']=="partial_then_full", "Only 'partial_then_full' quantization implemented atm"
  sample = 10 # How many linear iterations in between printing

  #extract commmonly used hyperparameters to nicer var names
  output_bits = hyp['output_bits']
  iters,qiter,batch = hyp['iters'],hyp['qiter'],hyp['batch']

  data = datasets.Mnistdata(image_width=hyp['image_width'],input_bits=1,X_format="-1,1",y_format="0,1")
  #data = datasets.Mnistdata(image_width=8,input_bits=2,X_format="float")
  #tdata = data.next_data(1)
  #print (tdata)
  #def __init__(self,image_width=28,input_bits=1,X_format="-1,1",y_format="0,1"):
  H = hyp['image_width']
  W = hyp['image_width']
  Xbits = H*W

  lut_bits = 4
  ybits = 10

  X = tf.placeholder(tf.float32, shape=[None,Xbits])
  y_ = tf.placeholder(tf.float32, shape=[None,ybits])
  X_reshape = tf.reshape(X,[-1,H,W,1])

  lnum=2
  lWs = [None for i in range(lnum)]
  l = [None for i in range(lnum)]

  l[0], lWs[0] = ConvLayer(N=4,Cout=30,filt=[7,7],stride=[2,2],padding="VALID")(X_reshape)
  print("L0",l[0])
  l[1], lWs[1] = ConvLayer(N=4,Cout=60,filt=[8,8],stride=[1,1],padding="VALID")(l[0])
  #l[2], lWs[2] = ConvLayer(N=4,Cout=32,filt=[5,5],stride=[1,1],padding="SAME")(l[1])
  #l[3], lWs[3] = ConvLayer(N=4,Cout=32,filt=[2,2],stride=[2,2],padding="SAME")(l[2])
  #l[4], lWs[4] = ConvLayer(N=4,Cout=60,filt=[5,5],stride=[1,1],padding="VALID")(l[3])
  print("LLLLL",l)
  y = l[1]
  yshape = y.get_shape().as_list()
  print("YSHAPE",yshape)
  assert yshape[1]*yshape[2]*yshape[3]==10*output_bits
  print("LAYERS")
  for lay in l:
    print("  ",lay)

  print (y)
  #scale = np.ones([1,1,output_bits])
  #scale[0][0] = np.array([1,1,1,1,1,1,1,1])
  y = tf.reshape(y,[-1,10,output_bits]) #* scale
  print (y)
  y = tf.reduce_sum(y,2)
  print (y)
  Ws = flatten_list(lWs)
  totLuts = 0
  print (len(Ws))
  for w in Ws:
    print(w)
    shape = w.get_shape().as_list()
    totLuts += shape[0]*shape[1]
  print("Total Luts", totLuts//2**lut_bits)

  W_quantized = [tf.placeholder(tf.float32,shape=w.shape) for w in Ws]
  W_assigns = [tf.assign(Ws[i],W_quantized[i]) for i in range(len(Ws))]

  loss_pre = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
  loss = loss_pre + binary_l1_reg(Ws,hyp['reg_weight_inner'],hyp['reg_weight_outer'])
  train_step = tf.train.AdamOptimizer(hyp['learning_rate']).minimize(loss)

  ymax = tf.reduce_max(y,axis=1)
  ymax = tf.reshape(ymax,[-1,1])
  ymax = tf.tile(ymax,[1,10])
  yscale = y >= ymax
  y_scale = tf.cast(y_,tf.bool)
  
  correct_pred = tf.cast(tf.reduce_all(tf.equal(yscale,y_scale),1),tf.float32)
  accuracy = tf.reduce_mean(correct_pred)

  losslog = np.zeros(((iters*qiter)//sample) + 1)
  hist = None
  luthists = None
  luthistdeps = None
  q_accuracy = np.zeros(qiter)
  uq_accuracy = np.zeros(qiter)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for j in range(qiter):
      for i in range(iters):
        tdata = data.next_data(batch)
        yval,lossval = None,None
        y1,y2 = None,None

        _,yval,lossval,y1,y2,acc = sess.run([train_step,y,loss,yscale, y_scale,accuracy],feed_dict={X:tdata[0],y_:tdata[1]})

        if (i%sample==0):
          print(lossval, j, "("+str(i)+"/"+str(iters)+")")
          print("  cor",tdata[1][0])
          print("  lrn",(yval[0]))
          print("  ysca",y1[0])
          print("  y_sca",y2[0])
          print("  acc",acc)
          losslog[(i+j*iters)//sample] = lossval
      print(j, "Accuracy_test")
      uq_accuracy[j] = accuracy.eval(feed_dict={X:data.test[0],y_:data.test[1]})
      print(uq_accuracy[j])


      curWs = sess.run(Ws,feed_dict={X:data.test[0],y_:data.test[1]})
      #hist = histogram(curWs)

      print("curW5",curWs[2])

      fd = make_feed_dict(W_quantized,curWs,hyp['partial_quant_threshold'],True)
      full_quant_fd = make_feed_dict(W_quantized,curWs)
      
      sess.run(W_assigns,feed_dict=full_quant_fd)
      QWs = sess.run(Ws,feed_dict={X:data.test[0],y_:data.test[1]})
      #luthists = histLut(QWs)
      #luthistdeps = histLutDeps(QWs)
      q_accuracy[j] = accuracy.eval(feed_dict={X:data.test[0],y_:data.test[1]})
      print(j, "Accuracy_test_q")
      print(q_accuracy[j])
      if j == 5 and hyp['early_out'] and q_accuracy[j] < 0.2:
        print("We don't seem to be learning :(   Earlying out to save time.")
        return max(q_accuracy),losslog,hist,uq_accuracy,q_accuracy

      if j >= 5 and (max(q_accuracy) == q_accuracy[-5]):
        print("We don't seem to be learning :(   Earlying out to save time.")
        return max(q_accuracy),losslog,hist,uq_accuracy,q_accuracy

      if not (j>int(hyp['quant_iter_threshold']*qiter)):
        sess.run(W_assigns,feed_dict=fd)

  print("Maximum quantized accuracy: " + str(max(q_accuracy)) + ", final: " + str(q_accuracy[-1]))
  if display_graphs:
    plt.figure(1)
    plt.plot(losslog)
    plt.xlabel("iter/"+str(sample))
    plt.ylabel("loss")
    plt.show()
    #plt.figure(2)
    #plt.plot(luthists)
    plt.figure(3)
    print(luthistdeps)
    plt.plot(luthistdeps)
    plt.show()
    plt.figure(4)
    plt.hist(hist,bins=100)
    plt.show()
    plt.figure(5)
    plt.plot(uq_accuracy)
    plt.plot(q_accuracy)
    plt.legend(['unquantized', 'quantized'], loc='upper left')
    plt.xlabel("qiter")
    plt.ylabel("accuracy")
    plt.show()
  return max(q_accuracy),losslog,hist,uq_accuracy,q_accuracy


if __name__ == '__main__':
  hyp = dict(
    image_width = 20,
    learning_rate = 0.01,
    reg_weight_inner = 0.00001,
    reg_weight_outer = 1.0,
    output_bits = 2,
    qiter = 3,
    iters = 300,
    batch = 32,
    quant_scheme = "partial_then_full",
    quant_iter_threshold = 0.75, # switchover 75% of the way through
    early_out = True,
    partial_quant_threshold = 0.95
  )
  #a = run_mnist(hyp,True)
  a = run_mnist_conv({'image_width': 21, 'learning_rate': 0.003, 'reg_weight_inner': 1.3161390636688435e-10, 'reg_weight_outer': 0.20233250452973095, 'output_bits': 6, 'qiter': 5, 'iters': 750, 'batch': 32, 'quant_scheme': 'partial_then_full', 'quant_iter_threshold': 0.9, 'early_out': True, 'partial_quant_threshold': 0.9341594717634678},False)
  print(a)

  
