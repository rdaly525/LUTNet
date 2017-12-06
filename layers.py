import numpy as np
import tensorflow as tf
import math
import sys

from common import *

def broadcast(tensor,shape):
  return tensor + tf.zeros(shape,dtype=tensor.dtype)

def Var(k):
  return tf.Variable(tf.random_normal(k,mean=0,stddev=0.5))

#Does not take lists!
def Mux(N,kind,sigma=1):
  if kind=="gaussian":
    assert(0)
    return MuxGaussian(N,sigma)
  if kind=="triangle":
    assert(0)
    return MuxTriangle(N)
  assert(0)

def check_mux_inputs(I,S,N):
  #one of I or S has to have two dimensions
  dI,dS=None,None
  assert type(I) is not list
  assert type(S) is not list
  dI = len(I.get_shape().as_list())
  dS = len(S.get_shape().as_list())
  #Verify that one of the inputs is just weights
  assert (dI==1 and dS==2) or (dI==2 and dS==1)
  if dI==1:
    I = tf.expand_dims(I,0)
  if dS==1:
    S = tf.expand_dims(S,0)
  
  #Verify that dimensions are correct
  assert I.get_shape().as_list()[1] == 2**N
  assert S.get_shape().as_list()[1] == N
  return I,S

def MuxGaussian(N,sigma):
  def mux(I,S):
    #I and S are now both 2 dimensions
    I,S = check_mux_inputs(I,S,N)
    def mv_norm(s,i):
      u = bitfield(i,N)
      #front = (1+1/math.e**(2.0/sigma))**(-N)
      front = 1.0
      snorm = s-u
      l2 = tf.reduce_sum(snorm*snorm,axis=1)
      return front*tf.exp(l2*(-0.5/sigma))
    
    norms = [mv_norm(S,i) for i in range(2**N)]
    norms_stack = tf.stack(norms,axis=1)
    outpre = tf.reduce_sum(norms_stack*I,axis=1)
    #return outpre
    return tf.tanh(outpre)
  return mux

#out = f(I,S)
#I.shape == [K,2**N]
#S.shape == [-1,K,N]
#out.shape == [-1,K]
def MuxSTriangle(N,K):
  def mux(I,S):
    Ishape = I.get_shape().as_list()
    Sshape = S.get_shape().as_list()
    assert len(Ishape) == 2
    assert Ishape[0] == K
    assert Ishape[1] == 2**N
    assert len(Sshape) == 3
    assert Sshape[1] == K
    assert Sshape[2] == N

    S0 = tf.maximum(0.0,1-tf.abs(S+1)/2.0)
    S1 = tf.maximum(0.0,1-tf.abs(S-1)/2.0)
    out = tf.expand_dims(I,0)
    for n in reversed(range(N)):
      mid = 2**n
      #print( "N", mid, out)
      out = out[:,:,0:mid]*S0[:,:,n:n+1] + out[:,:,mid:]*S1[:,:,n:n+1]
      #print( out)
    return out[:,:,0]
  return mux

#out = f(I,S)
#I.shape == [-1,K,2**N]
#S.shape == [K,N]
#out.shape == [-1,K]
def MuxITriangle(N,K):
  def mux(I,S):
    Ishape = I.get_shape().as_list()
    Sshape = S.get_shape().as_list()
    assert len(Ishape) == 3
    assert Ishape[1] == K
    assert Ishape[2] == 2**N
    assert len(Sshape) == 2
    assert Sshape[0] == K
    assert Sshape[1] == N

    S0 = tf.expand_dims(tf.maximum(0.0,1-tf.abs(S+1)/2.0),2)
    S1 = tf.expand_dims(tf.maximum(0.0,1-tf.abs(S-1)/2.0),2)
    out = I
    for n in reversed(range(N)):
      mid = 2**n
      out = out[:,:,0:mid]*S0[:,n] + out[:,:,mid:]*S1[:,n]
    return tf.out[:,:,0]
  return mux


def LutN(N,K,kind="triangle",sigma=1):
  def lut(x,W=None):
    if W is not None:
      Wshape = W.get_shape().as_list()
      assert len(Wshape) == 2
      assert Wshape[0] == K
      assert Wshape[1] == 2**N
      pass
    else:
      W = Var([K,2**N])
    out = MuxSTriangle(N,K)(W,x)
    return out,W
  return lut



#works best if:
#(K1*N)%K0==0
#out = LutLayer(x)
#in.shape = [-1,K0]
#out.shape = [-1,K1]
def LutLayer(N,K0,K1):
  def layer(x):
    xshape = x.get_shape().as_list();
    assert len(xshape) == 2
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
    out, Ws = LutN(N,K1)(x)
    return out, Ws
  return layer


#This is a sequence of LutLayers
#out = f(in)
#in.shape = [-1,Ls[0]]
#out.shape = [-1,Ls[-1]]
def MacroLutLayer(N,Ls):
  def layer(x):
    K0,K1 = Ls[0],Ls[-1]
    L = len(Ls)-1
    #This is the min number of layers needed in order to have the outputs depend on every input
    assert L >= math.ceil(log(N)(K0))
    
    Ws = [None for i in range(L)]
    l = x
    for i in range(L):
      print( "A", l, Ls[i], Ls[i+1])
      l, Ws[i] = LutLayer(N,Ls[i],Ls[i+1])(l)
    return l, Ws
  return layer


def SelectN(N,kind="gaussian",sigma=1):
  def sel(x,W=None):
    if W is not None:
      assert W.get_shape().as_list()[0] == N
      pass
    else:
      W = Var(N)
    if type(x) is list:
      x = tf.stack(x,axis=1)
    out = Mux(N,kind,sigma)(x,W)
    return out,W
  return sel


#Do not use K=1
#Pass in W at your own risk
def SelectK(K,kind="gaussian",sigma=1):
  def sel(x,W=None):
    if type(x) is list:
      x = tf.stack(x,axis=1)
    assert x.get_shape().as_list()[1]==K
    #base case
    if K==1:
      return x[:,0],W
    
    sbits = len(bitstr(K-1))
    if W is not None:
      #Need to get the bottom bits from W
      l = W.get_shape().as_list()[0] 
      if sbits != l:
        W = W[0:sbits]
      assert W.get_shape().as_list()[0] == sbits
    else:
      W = Var(sbits)
    if 2**sbits==K:
      out,_ = SelectN(sbits,kind,sigma)(x,W)
      return out,W
    else:
      mid = 2**(sbits-1)
      out0,_ = SelectN(sbits-1,kind,sigma)(x[:,0:mid],W[0:sbits-1])
      out1,_ = SelectK(K-mid,kind,sigma)(x[:,mid:],W[0:sbits-1])
      out,_ = SelectN(1,kind,sigma)([out0,out1],W[sbits-1:])
      return out,W
  return sel


def Selects(K,C,kind="gaussian",sigma=1):
  def layer(x):
    if type(x) is list:
      x = tf.stack(x,axis=1)
    assert x.get_shape().as_list()[1]==K
    select = SelectK(K,kind,sigma=sigma)
    outs = [None for c in range(C)]
    selWs = [None for c in range(C)]
    for c in range(C):
      outs[c], selWs[c] = select(x)
    return tf.stack(outs,axis=1),selWs
  return layer

#K inputs to each mux
#C LutNs
def SelectLutLayer(K,N,C,kind="gaussian",sigma=1):
  def layer(x):
    if type(x) is list:
      x = tf.stack(x,axis=1)
    
    assert x.get_shape().as_list()[1]==K
    Kup = 2**(len(bitstr(K)))
    if K!=Kup:
      c0 = tf.fill(tf.shape(x[:,0:(Kup-K)]),-1.0)
      x = tf.concat([x,c0],axis=1)
    assert x.get_shape().as_list()[1]==Kup

    selects = Selects(Kup,N,kind,sigma=sigma)
    lut = LutN(N,kind,sigma=sigma)
    selWs = [ [None for n in range(N)] for c in range(C)]
    outs = [None for c in range(C)]
    lutWs = [None for c in range(C)]
    for c in range(C):
      muxout, selWs[c] = selects(x)
      outs[c],lutWs[c] = lut(muxout)
    return tf.stack(outs,axis=1),selWs,lutWs
  return layer

def SelectLutLayers(N,inW,outW,L,kind="gaussian",sigma=1):
  def layers(x):
    if type(x) is list:
      x = tf.stack(x,axis=1)
    selWs = [None for l in range(L)]
    lutWs = [None for l in range(L)]
    curl = x
    curbits = inW
    for li in range(L):
      nextbits = int(outW + ((L-1-li)*1.0*(inW-outW))/(L))
      curl, selWs[li], lutWs[li] = SelectLutLayer(curbits,N,nextbits,kind,sigma)(curl)
      curbits = nextbits
    return curl, selWs,lutWs
  return layers

def binary_reg(W):
  if not type(W) is list:
    W = [W]
  ws = []
  for w in W:
    wm1 = w-1
    wp1 = w+1
    ws.append(tf.reduce_sum(wm1*wm1*wp1*wp1))
  return tf.add_n(ws)

def binary_l1_reg(W):
  if not type(W) is list:
    W = [W]
  ws = []
  for w in W:
    ws.append(tf.reduce_sum(tf.abs(1-tf.abs(w))))
  return tf.add_n(ws)



def randConnection(inW,outW):
  minNum = int(4*outW//inW)
  assert minNum > 0
  choices = [[i,minNum] for i in range(inW)]
  rerror = 4*outW-inW*minNum
  errorperm = np.random.permutation(np.array(range(inW)))
  for i in range(rerror):
    choices[errorperm[i]][1] +=1
  cons = np.zeros((outW,4)).astype(int)
  for i in range(outW):
    if len(choices) < 4 :
      print( choices)
      #Have to distribute the rest 
      #could have more than 1 i slot
      #TODO Should distribute evenly for the more than one slot case
      j=0
      di=0
      for choice in choices:
        for _ in range(choice[1]):
          di = j//4
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
  lutfun = LutN(N,"gaussian",sigma)

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
    print( str(inW)+"->"+str(outW),ri_stats)
    return outs,Ws
  return layer
    
def lutlayersimp(N,sigma,inW,outW):
  def adjust(X,w):
    bnum = tf.shape(X)[0]
    if (w%N !=0):
      adj = w//N
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
    print( X)
    print( [H,W,Cin])
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
