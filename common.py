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

def sacleto11(val,check=True):
  if type(val) is int:
    val = [val]
  if type(val) is list:
    val = np.array(val)
  assert type(val) is np.ndarray
  assert not check or np.all((val==0) + (val==1))
  return val*2-1

def scaleto01(val,check=True):
  if type(val) is int:
    val = [val]
  if type(val) is list:
    val = np.array(val)
  assert type(val) is np.ndarray
  assert not check or np.all((val==-1) + (val==1))
  return (val+1)/2


