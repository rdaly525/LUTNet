import numpy as np


def flatten_list(l):
  """ Single-level flatten """
  return [item for sublist in l for item in sublist]

def make_feed_dict(phs, curs,thresh=0,prob=1):
  """
  phs=placeholders
  curs=current values
  """
  assert len(phs) == len(curs)
  def quant(x):
    if (x >=thresh and np.random.uniform() < prob):
      return 1
    elif (x < -thresh and np.random.uniform() < prob):
      return -1
    else:
      return np.random.normal(0,0.5)
  quant = np.vectorize(quant,otypes=[np.float])
  news = [quant(cur) for cur in curs]
  feed_dict = {}
  for i in range(len(phs)):
    feed_dict[phs[i]] = news[i]
  return feed_dict

def bitstr(i,N=None):
  """ Convert integer i into string binary representation
      Optionally pads to N binary digits.
      String starts at LSB.

      Examples: bitstr(18) == "01001"
                bitstr(18,8) == "01001000"
  """
  ret = bin(i)[2:][::-1]
  if N:
    for _ in range(N-len(ret)):
      ret = ret+'0'
  return ret

def bitfield(i,N):
  """ Convert integer i into binary representation then converts bits into {-1.0,1.0}
      Optionally pads to N binary digits.
      Index 0 is LSB.
  """ 
  bits = bitstr(i,N)
  bits = [int(digit)*2-1 for digit in bits]
  return np.array(bits).astype(float)


def scaleto11(val,check=True):
  """Scales From [0,1] to [-1,1]"""
  if type(val) is not list and type(val) is not np.ndarray:
    val = [val]
  if type(val) is list:
    val = np.array(val)
  assert type(val) is np.ndarray
  assert not check or np.all((val==0) + (val==1))
  return val*2-1

def scaleto01(val,check=True):
  """Scales From [-1,1] to [0,1]"""
  if type(val) is not list and type(val) is not np.ndarray:
    val = [val]
  if type(val) is list:
    val = np.array(val)
  assert type(val) is np.ndarray
  assert not check or np.all((val==-1) + (val==1))
  return (val+1)/2.0


