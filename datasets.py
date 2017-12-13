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
  def __init__(self,train,test,data=None):
    assert isinstance(train,Data)
    assert isinstance(test,Data)
    self.train_data = train
    self.test_data = test
    self.data = data

class Mnistdata(Dataset):
  def __init__(self,image_width=28,input_bits=1,X_format="-1,1",y_format="0,1"):
    assert image_width in range(1,29)
    assert input_bits in range(1,9)
    assert X_format in ("-1,1","0,1","int")
    assert y_format in ("-1,1","0,1")

    self.image_width = image_width
    self.input_bits =input_bits 
    self.bit_mask = sum([2**(7-i) for i in range(input_bits)])
    self.X_format = X_format
    self.y_format = y_format
    data = mnist.input_data.read_data_sets('MNIST_data',one_hot=True)
    train_images = self.reformatX(data.train.images)
    train_labels = self.reformatY(data.train.labels)
    test_images = self.reformatX(data.test.images)
    test_labels = self.reformatY(data.test.labels)
    Dataset.__init__(self,Data(train_images,train_labels),Data(test_images,test_labels),data)

  def downsample(self,X):
    assert len(X.shape)==2
    bsize = X.shape[0]
    W = self.image_width
    X = X.reshape((bsize,W,W))
    newX = np.zeros((len(X), W, W), dtype=np.float)
    for i in range(len(X)):
      orig_im = Image.fromarray(X[i],'F')
      new_im = orig_im.resize((W,W), Image.LANCZOS)
      newX[i] = np.asarray(new_im)
      #new_im.save("temp.tiff", "TIFF")
    return X.reshape((bsize,W*W))

  #assumes 28x28 in "int" format
  def reformatX(self,X):
    #Downsample
    X = self.downsample(X)
    #Mask out unneeded bits
    X = (255*X).astype(int)
    if self.X_format == "int":
      return X & self.bit_mask
    if (self.input_bits == 1):
      X = (X >= 2**7).astype(int)
      if self.X_format == "-1,1":
        return scaleto11(X)

    assert 0

  #assumes 0,1 format
  def reformatY(self,y):
    if self.y_format == "-1,1":
      return scaleto11(y)
    return y

  def next_data(self,k):
    data = self.data.train.next_batch(k)
    return [self.reformatX(data[0]), self.reformatY(data[1]) ]

  @property
  def test(self):
    return [self.test_data.inputs,self.test_data.outputs]

  @property
  def train(self):
    return [self.train_data.inputs,self.train_data.outputs]
 


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


     











