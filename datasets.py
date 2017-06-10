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


class binopdata:
  def __init__(self,N,f,inbits,outbits):
    self.N = N
    self.f = f
    self.inbits = inbits
    self.outbits = outbits

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
  def test_data(self):
    inbits = self.inbits
    outbits = self.outbits
    inbitrange = 2**inbits
    X,y = np.zeros((inbitrange**2,2*inbits)),np.zeros((inbitrange**2,outbits))
    for a in range(inbitrange):
      for b in range(inbitrange):
        c = self.f(a,b)
        i = a*(2**inbits)+b
        X[i,0:inbits] = bitfield(a,inbits)
        X[i,inbits:] = bitfield(b,inbits)
        y[i] = bitfield(c,outbits)
    return [X,y]

