import numpy as np
from common import *

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

class Lutdata(Dataset):
  def __init__(self,N,f):
    self.N = N
    self.f = f
    data = np.zeros(2**N).astype(int)
    for i in range(2**N):
      data[i] = f(i,N)
    Dataset.__init__(self,data,data)

  def next_data(self,k):
    X,y = np.zeros((k,self.N)),np.zeros(k)
    for i in range(k):
      ri = np.random.randint(0,2**self.N)
      X[i], y[i] = bitfield(ri,self.N), scaleto11(self.f(ri,self.N))
    return [X,y]

  @property
  def test(self):
    return self.test

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
    return self.test

class Mnistdata(Dataset):
  def __init__(self,ds=1):
    assert 28%ds==0
    self.ds = ds
    self.data =  mnist.input_data.read_data_sets('MNIST_data',one_hot=True)
    train_images = (self.data.train.images >0.5).astype(int)
    train_labels = (self.data.train.labels > 0.5).astype(int)
    test_images = (self.data.test.images > 0.5).astype(int)
    test_labels = (self.data.test.labels > 0.5).astype(int)
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
    return X[:,::self.ds,::self.ds].reshape(X.shape[0],(28/self.ds)**2)

  def next_data(self,k):
    data = self.data.train.next_batch(k)
    data = self.downsample(data)
    data[0] = scaleto11((data[0] > 0.5).astype(int))
    data[1] = scaleto11((data[1] > 0.5).astype(int))
    return data

  @property
  def test(self,ds=1):
    return [self.downsample(scaleto11((self.test_data.inputs > 0.5).astype(int))),scaleto11((self.test_data.outputs >0.5).astype(int))]











