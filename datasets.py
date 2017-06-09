import numpy as np
from common import *

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

