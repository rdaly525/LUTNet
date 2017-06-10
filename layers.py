import numpy as np
import tensorflow as tf
import math
import sys

from common import *

#def lutN1(N,sigma):
#  assert sigma==1
#  def lut(x):
#    def mv_norm(x,i):
#      u = bitfield(i,N)
#      #Little sketchy math to figure out the value .792. 
#      front = math.exp(.792*N)*(((2*math.pi)**N))**(-0.5)
#      xnorm = x-u
#      l2 = tf.reduce_sum(xnorm*xnorm,axis=1)
#      return front*tf.exp(l2*(-1/(2.0)))
#    
#    assert x.shape[1]==N
#    w = tf.Variable(tf.random_normal([2**N]))
#    norms = [mv_norm(x,i) for i in range(2**N)]
#    norms_stack = tf.stack(norms,axis=1)
#    outpre = tf.reduce_sum(norms_stack*w,axis=1)
#    return tf.tanh(outpre),w
#  return lut

def lutN(N,sigma):
  def lut(x):
    def mv_norm(x,i):
      u = bitfield(i,N)
      front = (1+1/math.e**(2.0/sigma))**(-N)
      xnorm = x-u
      l2 = tf.reduce_sum(xnorm*xnorm,axis=1)
      return front*tf.exp(l2*(-0.5/sigma))
    
    assert x.shape[1]==N
    w = tf.Variable(tf.random_normal([2**N],mean=0,stddev=0.5))
    norms = [mv_norm(x,i) for i in range(2**N)]
    norms_stack = tf.stack(norms,axis=1)
    outpre = tf.reduce_sum(norms_stack*w,axis=1)
    return tf.tanh(outpre),w
  return lut

def binary_reg(W):
  if not type(W) is list:
    W = [W]
  ws = []
  for w in W:
    wm1 = w-1
    wp1 = w+1
    ws.append(tf.reduce_sum(wm1*wm1*wp1*wp1))
  return tf.add_n(ws)

def randConnection(inW,outW):
  minNum = int(4*outW/inW)
  assert minNum > 0
  choices = [[i,minNum] for i in range(inW)]
  rerror = 4*outW-inW*minNum
  errorperm = np.random.permutation(np.array(range(inW)))
  for i in range(rerror):
    choices[errorperm[i]][1] +=1
  cons = np.zeros((outW,4)).astype(int)
  for i in range(outW):
    if len(choices) < 4 :
      #Have to distribute the rest 
      #could have more than 1 i slot
      j=0
      di=0
      for choice in choices:
        for _ in range(choice[1]):
          di = j/4
          cons[i+di][j%4] = choice[0]
          j +=1
      assert j%4==0
    else:
      indices = np.random.choice(len(choices),4,replace=False)
      indices.sort() 
      indices = np.flip(indices,0) #Needed to delete properly
      for j in range(4):
        idx = indices[j]
        cons[i][j] = choices[idx][0]
        if choices[idx][1]==1:
          del choices[idx]
        else:
          choices[idx][1] -= 1
  return np.random.permutation(cons)

def lutlayer(N,sigma,inW,outW):
  assert outW >=4
  lutfun = lutN(N,sigma)

  def layer(X):
    assert X.shape[1] == inW
    #pick 4*outW random indices of from inW
    Ws = []
    layer_outputs= []
    cons = randConnection(inW,outW)
    ri_stats = [0 for i in range(inW)]
    for i in range(outW):
      lut_inputs = []
      for j in range(4):
        ri = cons[i][j]
        ri_stats[ri] += 1
        lut_inputs.append(X[:,ri])
      lut_ins = tf.stack(lut_inputs,axis=1)
      lut_out, W = lutfun(lut_ins)
      layer_outputs.append(lut_out)
      Ws.append(W)
    outs = tf.stack(layer_outputs,axis=1)
    print ri_stats
    return outs,Ws
  return layer
    

#N bits
#H,W initial height and width
#Cin is input channels
#Cout is output channels
def lutConvlayer(N,H,W,Cin,Cout):
  assert Cin==4
  lut = lutN(N,1)
  def layer(X):
    print X
    print [H,W,Cin]
    assert X.shape[1:] == [H,W,Cin]
    Ws = []
    luth = []
    for h in range(H):
      Xh = X[:,h]
      lutw = []
      for w in range(W):
        Xhw = Xh[:,w]
        lutc = []
        for c in range(Cout):
          out, weight = lut(Xhw)
          Ws.append(weight)
          lutc.append(out)
        lutw.append(tf.stack(lutc,axis=1))
      luth.append(tf.stack(lutw,axis=1))
    return tf.stack(luth,axis=1), Ws
  return layer
