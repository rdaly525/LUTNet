import random
import math
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import heapq
import sklearn.decomposition as sc
from sklearn.utils.extmath import randomized_svd
from math import sqrt
from scipy.linalg import eig
from PIL import Image
import scipy.misc

if __name__ == "__main__":


  #Use regularization: reg(x) = 1-2x**2 + x**4
  def bitfield(i,N):
    bits = [int(digit) for digit in bin(i)[2:]]
    for i in range(N-len(bits)):
      bits.insert(0,0)
    #change to -1 to 0
    return [bits[j]*2-1 for j in range(N)]

  def lutN(N,sigma):
    def norm(u):
      front = (1+1/math.e**(2.0/sigma))**(-N)
      def f(x):
        l2 = sum([(x[i]-u[i])**2 for i in range(N)])
        return front * math.exp(-l2/(2.0*sigma))
      return f


    def lut(init):
      norms = [norm(bitfield(i,N)) for i in range(2**N)]
      def run(x):
        assert len(x) == N
        ns = [norms[i](x) for i in range(2**N)]
        vals = [norms[i](x)*init[i] for i in range(2**N)]
        return sum(vals)
      return run
    return lut

  #l = lut([np.random.uniform(-1,1) for i in range(8)])
  for n in range(1,8):
    l = lutN(n,3.5)([1 for i in range(2**n)])
    print n, l([-1 for i in range(n)])
