import math
import numpy as np

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
  n = 2
  hcs = 0.82050897
  for i in range(11):
    x,y = np.random.uniform(-1,1,size=2)
    l = lutN(2,1)([1 for j in range(2**n)])
    print 1.0*i/10,l([1.0*i/10,1])



