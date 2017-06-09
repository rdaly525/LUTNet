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
    w = tf.Variable(tf.random_normal([2**N]))
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


#N bits
#H,W initial height and width
#Cin is input channels
#Cout is output channels
def lutlayerdepth(N,H,W,Cin,Cout):
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

def lutlayerspatial():
  assert(0)

def lutlayerpool():
  assert(0)

def lutlayer(N,H,W,Cin,Cout,ltype):
  assert N==4
  assert Cin==4 or Cin==1
  if ltype=="depth":
    return lutlayerdepth(N,H,W,Cin,Cout)
  elif ltype=="spatial":
    return lutlayerspatial(N,H,W,Cin,Cout)
  elif ltype=="pool":
    return lutlayerpool(N,H,W,Cin,Cout)
  assert(0)





