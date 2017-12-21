import numpy as np
import tensorflow as tf
import math
import sys

from common import *

def broadcast(tensor,shape):
  return tensor + tf.zeros(shape,dtype=tensor.dtype)

def Var(k,stddev=0.5):
  return tf.Variable(tf.random_normal(k,mean=0,stddev=stddev))


#out = f(I,S)
#I.shape == [K,2**N]
#S.shape == [-1 (,WH), K,N] 
#out.shape == [-1 (,WH), K]
#Note: This function can either take in 3 or 4 dimensions for the S param.
#      The WH is used when doing a convolution

def MuxSTriangle(N,K):
  def mux(I,S):
    Ishape = I.get_shape().as_list()
    Sshape = S.get_shape().as_list()
    assert len(Ishape) == 2
    assert Ishape[0] == K
    assert Ishape[1] == 2**N
    assert len(Sshape) in (3,4)
    S0 = tf.maximum(0.0,1-tf.abs(S+1)/2.0)
    S1 = tf.maximum(0.0,1-tf.abs(S-1)/2.0)
    
    if len(Sshape) == 3: 
      assert Sshape[1] == K
      assert Sshape[2] == N

      out = tf.expand_dims(I,0)
      for n in reversed(range(N)):
        mid = 2**n
        out = out[:,:,0:mid]*S0[:,:,n:n+1] + out[:,:,mid:]*S1[:,:,n:n+1]
      return out[:,:,0]
    else: #Used for convolutions
      assert len(Sshape) == 4
      assert Sshape[2] == K
      assert Sshape[3] == N

      out = tf.expand_dims(tf.expand_dims(I,0),0)
      for n in reversed(range(N)):
        mid = 2**n
        out = out[:,:,:,0:mid]*S0[:,:,:,n:n+1] + out[:,:,:,mid:]*S1[:,:,:,n:n+1]
      return out[:,:,:,0]
  return mux

def LutN(N,K,stddev=0.5):
  def lut(x,W=None):
    if W is not None:
      Wshape = W.get_shape().as_list()
      assert len(Wshape) == 2
      assert Wshape[0] == K
      assert Wshape[1] == 2**N
    else:
      W = Var([K,2**N],stddev)
    out = MuxSTriangle(N,K)(W,x)
    return out,W
  return lut

#works best if:
#(K1*N)%K0==0
#out = LutLayer(x)
#in.shape = [-1 (WH,), K0]
#out.shape = [-1 (WH,), K1]
def LutLayer(N,K0,K1,stddev=0.5):
  def layer(x,W=None):
    xshape = x.get_shape().as_list();
    
    if len(xshape)==2:
      assert xshape[1] == K0
      
      mulfac = N*K1//K0
      if mulfac > 1:
        x = tf.tile(x,[1,mulfac])
      print( x)
      #Adjust input to a multiple of K1xN
      kmod = (K1*N)%K0
      if kmod != 0:
        x = tf.concat([x,x[:,0:kmod]],axis=1)
      assert x.get_shape().as_list()[1]==K1*N
      x = tf.reshape(x,[-1,K1,N])
      out, Ws = LutN(N,K1,stddev)(x,W)
      return out, Ws
    else: # Used for convolutions
      print(xshape)
      assert len(xshape)==3
      assert xshape[2] == K0
      WH = xshape[1]
      mulfac = N*K1//K0
      if mulfac > 1:
        x = tf.tile(x,[1,1,mulfac])
      print(x)
      #Adjust input to a multiple of K1xN
      kmod = (K1*N)%K0
      if kmod != 0:
        x = tf.concat([x,x[:,:,0:kmod]],axis=2)
      assert x.get_shape().as_list()[2]==K1*N
      x = tf.reshape(x,[-1,WH,K1,N])
      out, Ws = LutN(N,K1,stddev)(x,W)
      return out, Ws
  return layer

#This is a sequence of LutLayers
#out = f(in)
#in.shape = [-1,Ls[0]]
#out.shape = [-1,Ls[-1]]
def MacroLutLayer(N,Ls,stddev=0.5):
  def layer(x,Ws=None):
    K0,K1 = Ls[0],Ls[-1]
    L = len(Ls)-1
    #This is the min number of layers needed in order to have the outputs depend on every input
    assert L >= math.ceil(log(N)(K0))
    
    if Ws:
      newWs = Ws
    else:
      newWs = [None for i in range(L)]
    assert len(newWs) == L
    l = x
    for i in range(L):
      print( "A", l, Ls[i], Ls[i+1])
      l, newWs[i] = LutLayer(N,Ls[i],Ls[i+1],stddev)(l,newWs[i])
    return l, newWs
  return layer

def SingleMacroLayer(N,K0,K1,stddev=0.5):
  steps = math.ceil(log(N)(K0))
  Ls = create_layers(K0,K1,steps)
  print (Ls)
  return MacroLutLayer(N,Ls,stddev)

#N bits
#Cout is output channels
#filt is filter hieght and width
#stride is the stride in x and y

#X.shape =  (-1,H,W,Cin)
#out.shape = (-1,newH,newW,Cout)
#Ws = SingleMacroLayer(N,fh*fw*Cin,Cout)
def ConvLayer(N,Cout,filt=[3,3],stride=[1,1],stddev=0.5):
  assert len(filt) == 2
  assert len(stride) == 2
  fh,fw = filt
  sh,sw = stride
  K1 = Cout
  def layer(x):
    xshape = x.get_shape().as_list();
    print(xshape)
    assert len(xshape) == 4
    H,W,Cin = x.get_shape().as_list()[1:]
    #Verifies no padding
    print("(%d-%d)%%%d == 0?" % (H,fh,sh))
    assert (H-fh)%sh == 0
    assert (W-fw)%sw == 0

    K0 = fh*fw*Cin
    
    #This awesome function basically is an im2col paramterized by:
    #patch size (ksizes), and strides
    xpatch = tf.extract_image_patches(
        x,
        ksizes=[1,filt[0],filt[1],1],
        strides=[1,stride[0],stride[1],1],
        rates=[1,1,1,1],
        padding="VALID"
    )
    print ("xpat",xpatch)
    pshape = xpatch.get_shape().as_list()
    xpatch_flat = tf.reshape(xpatch,[-1,pshape[1]*pshape[2],pshape[3]])
    out_flat, Ws = SingleMacroLayer(N,K0,K1,stddev=stddev)(xpatch_flat)
    out = tf.reshape(out_flat,[-1,pshape[1],pshape[2],K1])
    return out,Ws
  return layer

def binary_reg(W):
  if not type(W) is list:
    W = [W]
  ws = []
  for w in W:
    wm1 = w-1
    wp1 = w+1
    ws.append(tf.reduce_sum(wm1*wm1*wp1*wp1))
  return tf.add_n(ws)

def binary_l1_reg(W,inner=0.00001,outer=1):
  if not type(W) is list:
    W = [W]
  winner = []
  wouter = []
  for w in W:
    winner.append(tf.reduce_sum(tf.maximum(0.0,1.0-tf.abs(w))))
    wouter.append(tf.reduce_sum(tf.maximum(0.0,tf.abs(w)-1.0)))
  return inner*tf.add_n(winner) + outer*tf.add_n(wouter)
