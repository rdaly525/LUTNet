import numpy as np
from common import *

class lutdata:
  def __init__(self,N,f):
    self.N = N
    self.f = f
    self.lutref = np.zeros(2**N).astype(int)
    for i in range(2**N):
      self.lutref[i] = f(i,N)


  def next_data(self,k):
    X,y = np.zeros((k,self.N)),np.zeros(k)
    for i in range(k):
      ri = np.random.randint(0,2**self.N)
      X[i], y[i] = bitfield(ri,self.N), scaleto11(self.f(ri,self.N))
    return [X,y]

  @property
  def correct(self):
    return self.lutref


class adddata:
  def __init__(self,N,bits):
    self.N = N
    self.bits = bits

  def next_data(self,k):
    bits = self.bits
    X,y = np.zeros((k,2*bits)),np.zeros((k,bits+1))
    for i in range(k):
      a,b = np.random.randint(0,2**bits,2)
      plus = a+b
      X[i,0:bits] = bitfield(a,bits)
      X[i,bits:] = bitfield(b,bits)
      y[i] = bitfield(plus,bits+1)
    return [X,y]
