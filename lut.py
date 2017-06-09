import numpy as np
import tensorflow as tf
import math

if __name__ == '__main__':
  def bitstr(i,N):
    ret = bin(i)[2:]
    for _ in range(N-len(ret)):
      ret = '0'+ret
    return ret
  def bitfield(i,N):
    bits = bitstr(i,N)
    bits = [int(digit)*2-1 for digit in bits]
    return np.array(bits).astype(float)
      
  class lutdata:
    def __init__(self,N,f):
      self.N = N
      self.f = f
      self.lutref = np.zeros(2**N).astype(int)
      for i in range(2**N):
        self.lutref[i] = f(i,N)

    def scale(self,val):
      assert type(val)==int
      assert val==0 or val==1
      return val*2-1

    def next_data(self,k):
      X,y = np.zeros((k,self.N)),np.zeros(k)
      for i in range(k):
        ri = np.random.randint(0,2**self.N)
        X[i], y[i] = bitfield(ri,self.N), self.scale(self.f(ri,self.N))
      return [X,y]
  
    @property
    def correct(self):
      return self.lutref

  def lutN(N,sigma):
    assert sigma==1
    def lut(x):
      def mv_norm(x,i):
        with tf.name_scope("mv_norm"+str(i)):
          u = bitfield(i,N)
          s2 = sigma**2
          #Little sketchy math to figure out the value .792. 
          front = math.exp(.792*N)*(((2*math.pi)**N)*s2)**(-0.5)
          xnorm = x-u
          l2 = tf.reduce_sum(xnorm*xnorm,axis=1)
        return front*tf.exp(l2*(-1/(2.0*s2)))
      
      assert x.shape[1]==N
      with tf.name_scope("lut"+str(N)):
        with tf.name_scope("weights"):
          w = tf.Variable(tf.random_normal([2**N]))
          #w = tf.Variable(tf.ones(2**N))
        norms = [mv_norm(x,i) for i in range(2**N)]
        norms_stack = tf.stack(norms,axis=1)
        outpre = tf.reduce_sum(norms_stack*w,axis=1)
        return tf.tanh(outpre),w
    return lut
  
  def reg(W):
    Wm1 = W-1
    Wp1 = W+1
    return tf.reduce_sum(Wm1*Wm1*Wp1*Wp1)

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


  N = 5
  data = lutdata(N,mod5)
  lr = 0.1
  rw = 0.01
  x = tf.placeholder(tf.float32, shape=[None,N])
  y_ = tf.placeholder(tf.float32, shape=[None])
  y, W = lutN(N,1)(x)
  loss = tf.nn.l2_loss(y-y_) + rw*reg(W)
  #loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  
  def descW(W):
    dw = (W > 0).astype(int)
    return dw

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter('logs',sess.graph)
    wval = None
    for i in range(1000):
      tdata = data.next_data(4)
      _,wval,yval,lossval = sess.run([train_step,W,y,loss],feed_dict={x:tdata[0],y_:tdata[1]})
      print lossval, yval,tdata[1]
      #print "  ",wval,tdata[0]
    print wval
    print "lrnd", descW(wval)
    print "corr", data.correct
    writer.close()

