import numpy as np

def bitstr(i,N):
  ret = bin(i)[2:]
  for _ in range(N-len(ret)):
    ret = '0'+ret
  return ret
def bitfield(i,N):
  bits = bitstr(i,N)
  bits = [int(digit)*2-1 for digit in bits]
  return np.array(bits).astype(float)

