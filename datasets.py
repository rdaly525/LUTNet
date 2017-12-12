import numpy as np
from common import *

from PIL import Image

import tensorflow.examples.tutorials.mnist as mnist 

class Data:
  def __init__(self,inputs,outputs):
    assert type(inputs) is np.ndarray
    assert type(outputs) is np.ndarray
    self.inputs = inputs
    self.outputs = outputs

class Dataset:
  def __init__(self,train,test):
    assert isinstance(train,Data)
    assert isinstance(test,Data)
    self.train_data = train
    self.test_data = test

#K is number of mux inputs
class Seldata(Dataset):
  def __init__(self,K,selval):
    assert selval < K
    assert K < 20
    self.cnt = 0
    self.K = K
    self.selval = selval
    X,y = np.zeros((2**self.K,self.K)), np.zeros((2**self.K))
    for i in range(2**self.K):
      X[i] = bitfield(i,self.K)
      y[i] = self.sel(i)
    Dataset.__init__(self,Data(X,y),Data(X,y))
  def sel(self,x):
    return int(bitstr(x,self.K)[self.selval])
  
  def next_data(self,k, to11=True):
    X,y = np.zeros((k,self.K)), np.zeros((k))
    for i in range(k):
      ri = np.random.randint(0,2**self.K)
      if (to11):
        X[i], y[i] = bitfield(ri,self.K), scaleto11(self.sel(ri))
      else:
        X[i], y[i] = bitfield(ri,self.K), self.sel(ri)

    return [X,y]
      
    
    #max_i = 2**self.K
    #r = np.random.randint(0,max_i)
    #X,y = np.zeros((k,self.K)), np.zeros((k))
    #
    #if r+k < max_i:
    #  X = self.train_data.inputs[r:r+k]
    #  y = self.train_data.outputs[r:r+k]
    #else:
      

  @property
  def test(self, to11=True):
    return self.test_data

class Lutdata(Dataset):
  def __init__(self,N,f):
    self.N = N
    self.f = f
    data = np.zeros(2**N).astype(int)
    X,y = np.zeros((2**N,N)), np.zeros((2**N))
    for i in range(2**N):
      X[i] = bitfield(i,N)
      y[i] = f(i,N)
    Dataset.__init__(self,Data(X,y),Data(X,y))

  def next_data(self,k):
    X,y = np.zeros((k,self.N)),np.zeros((k))
    for i in range(k):
      ri = np.random.randint(0,2**self.N)
      X[i], y[i] = bitfield(ri,self.N), scaleto11(self.f(ri,self.N))
    return [X,y]

  @property
  def test(self):
    return self.test_data

class Unaryopdata(Dataset):
  def __init__(self,f,inbits,outbits):
    self.cnt = 0
    self.f = f
    self.inbits = inbits
    self.outbits = outbits
    inbitrange = 2**inbits
    X,y = np.zeros((inbitrange,inbits)),np.zeros((inbitrange,outbits))
    for a in range(inbitrange):
      c = self.f(a)
      i = a
      X[i] = bitfield(a,inbits)
      y[i] = bitfield(c,outbits)
    Dataset.__init__(self,Data(X,y),Data(X,y))

  def next_data(self,k,rand=True):
    inbits = self.inbits
    outbits = self.outbits
    X,y = np.zeros((k,inbits)),np.zeros((k,outbits))
    if rand:
      for i in range(k):
        a = np.random.randint(0,2**inbits)
        c = self.f(a)
        X[i] = bitfield(a,inbits)
        y[i] = bitfield(c,outbits)
    else:
      tot = 2**inbits
      for i in range(k):
        a = (i+self.cnt)%tot
        c = self.f(a)
        X[i] = bitfield(a,inbits)
        y[i] = bitfield(c,outbits)
      self.cnt = (self.cnt+k)%tot
    return [X,y]
      
  @property
  def test(self):
    return self.test_data

class Binopdata(Dataset):
  def __init__(self,f,inbits,outbits):
    self.f = f
    self.inbits = inbits
    self.outbits = outbits
    inbitrange = 2**inbits
    X,y = np.zeros((inbitrange**2,2*inbits)),np.zeros((inbitrange**2,outbits))
    for a in range(inbitrange):
      for b in range(inbitrange):
        c = self.f(a,b)
        i = a*(2**inbits)+b
        X[i,0:inbits] = bitfield(a,inbits)
        X[i,inbits:] = bitfield(b,inbits)
        y[i] = bitfield(c,outbits)
    Dataset.__init__(self,Data(X,y),Data(X,y))

  def next_data(self,k):
    inbits = self.inbits
    outbits = self.outbits
    X,y = np.zeros((k,2*inbits)),np.zeros((k,outbits))
    for i in range(k):
      a,b = np.random.randint(0,2**inbits,2)
      c = self.f(a,b)
      X[i,0:inbits] = bitfield(a,inbits)
      X[i,inbits:] = bitfield(b,inbits)
      y[i] = bitfield(c,outbits)
    return [X,y]

  @property
  def test(self):
    return self.test_data



class Mnistdata(Dataset):
  def __init__(self,image_width=28,bit_depth=1,map_0_to_n1=True):
    # TODO: figure out what to do with bit_depth and map_0_to_n1
    assert bit_depth == 1
    assert map_0_to_n1 
    assert image_width<=28
    self.image_width = image_width
    self.bit_depth = bit_depth
    self.map_0_to_n1 = map_0_to_n1
    data = mnist.input_data.read_data_sets('MNIST_data',one_hot=True)
    train_images = data.train.images
    train_labels = data.train.labels
    test_images = data.test.images
    test_labels = data.test.labels
    Dataset.__init__(self,Data(train_images,train_labels),Data(test_images,test_labels))

  #assume [?,784]
  def reshape(self,X):
    bsize = X.shape[0]
    assert X.shape[1] == 28**2
    return np.reshape(X,(bsize,28,28))

  def downsample(self,X):
    if type(X) is tuple or type(X) is list:
      return [self.downsample(X[0]),X[1]]
    if len(X.shape)==2:
      X = self.reshape(X)

    W = self.image_width

    newX = np.zeros((len(X), W, W), dtype=np.float)
    for i in range(len(X)):
      orig_im = Image.fromarray(X[i],'F')
      new_im = orig_im.resize((W,W), Image.LANCZOS)
      newX[i] = np.asarray(new_im)
      #new_im.save("temp.tiff", "TIFF")
    return newX.reshape(newX.shape[0],W*W)

  def next_data(self,k):
    data = self.data.train.next_batch(k)
    data = self.downsample(data)
    data[0] = scaleto11((data[0] > 0.5).astype(int))
    data[1] = scaleto11((data[1] > 0.5).astype(int))
    return data

  #@property
  def test(self,to11=True):
    if (to11):
      return [scaleto11(( self.downsample(self.test_data.inputs) > 0.5).astype(int)),scaleto11((self.test_data.outputs >0.5).astype(int))]
    else:
      return [scaleto11(( self.downsample(self.test_data.inputs) > 0.5).astype(int)),(self.test_data.outputs >0.5).astype(int)]
  
  #@property
  def train(self, to11=True):
    if (to11):
      return [scaleto11(( self.downsample(self.train_data.inputs) > 0.5).astype(int)),scaleto11((self.train_data.outputs >0.5).astype(int))]
    else:
      return [scaleto11(( self.downsample(self.train_data.inputs) > 0.5).astype(int)),(self.train_data.outputs >0.5).astype(int)]
      











