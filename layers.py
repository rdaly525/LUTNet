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

def XlutN(N,sigma):
  def lut(x):
    def mv_norm(x,i):
      u = bitfield(i,N)
      front = (1+1/math.e**(2.0/sigma))**(-N)
      xnorm = x-u
      l2 = tf.reduce_sum(xnorm*xnorm,axis=1)
      return front*tf.exp(l2*(-0.5/sigma))

    if type(x) is list:
      x = tf.stack(x,axis=1)
    assert x.shape[1]==N
    w = tf.Variable(tf.random_normal([2**N],mean=0,stddev=0.5))
    norms = [mv_norm(x,i) for i in range(2**N)]
    norms_stack = tf.stack(norms,axis=1)
    outpre = tf.reduce_sum(norms_stack*w,axis=1)
    return tf.tanh(outpre),w
  return lut

def lutN(N,sigma=None):
  def bits(n):
    bstr = bitstr(n,N)
    return [int(digit) for digit in bstr]
  def lut(x):
    Xs = [[],[]]
    if type(x) is not list:
      x = [x[:,i] for i in range(N)]
    for n in range(N):
      Xs[0].append(tf.expand_dims(tf.clip_by_value(1-tf.abs(x[n]+1)/2.0,0.0,1.0),1))
      Xs[1].append(tf.expand_dims(tf.clip_by_value(1-tf.abs(x[n]-1)/2.0,0.0,1.0),1))
    w = tf.Variable(tf.random_normal([2**N],mean=0,stddev=0.5))
    out = tf.tile(tf.reshape(w,[1,2**N]),[tf.shape(Xs[0][0])[0],1])
    for n in range(N):
      mid = 2**(N-n-1)
      out = out[:,0:mid]*Xs[0][n] + out[:,mid:]*Xs[1][n]
    return tf.squeeze(out),w
    #out = None
    #if type(x) is list:
    #  out = tf.fill(tf.shape(x[0]),0.0)
    #else:
    #  out = tf.fill(tf.shape(x[:,0]),0.0)
    #for i in range(2**N):
    #  bs = bits(i)
    #  l = w[i]
    #  for n in range(N):
    #    l = l*Xs[bs[n]][n]
    #  out = out + l
    #return out,w
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
      print choices
      #Have to distribute the rest 
      #could have more than 1 i slot
      #TODO Should distribute evenly for the more than one slot case
      j=0
      di=0
      for choice in choices:
        for _ in range(choice[1]):
          di = j/4
          cons[i+di][j%4] = choice[0]
          j +=1
      assert j%4==0
      break
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

def lutlayerrand(N,sigma,inW,outW):
  assert outW >=1
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
    print str(inW)+"->"+str(outW),ri_stats
    return outs,Ws
  return layer
    
def lutlayersimp(N,sigma,inW,outW):
  def adjust(X,w):
    bnum = tf.shape(X)[0]
    if (w%N !=0):
      adj = w/N
      X = tf.concat([X,tf.fill([bnum,adj],-1.0)])

  def layers(X):
    assert X.shape[1] == inW
    X = adjust(X,inW)
    layer_outputs= []
    for o in range(outW):
      while(True):
        X.append


def lutlayers(N,sigma,inW,outW,L):
  def layers(X):
    assert X.shape[1] == inW

    Ws = []
    curl = X
    curbits = inW
    for li in range(L):
      nextbits = int(outW + ((L-1-li)*1.0*(inW-outW))/(L))
      curl, Wsl = lutlayerrand(N,sigma,curbits,nextbits)(curl)
      Ws += Wsl
      curbits = nextbits
    return curl, Ws
  return layers

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
