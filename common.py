import numpy as np


def flatten_list(l):
  return [item for sublist in l for item in sublist]

#phs=placeholders
#curs=current values
def make_feed_dict(phs, curs,thresh=0,prob=1):
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
  ret = bin(i)[2:][::-1]
  if N:
    for _ in range(N-len(ret)):
      ret = ret+'0'
  return ret

def bitfield(i,N):
  bits = bitstr(i,N)
  bits = [int(digit)*2-1 for digit in bits]
  return np.array(bits).astype(float)

def scaleto11(val,check=True):
  if type(val) is not list and type(val) is not np.ndarray:
    val = [val]
  if type(val) is list:
    val = np.array(val)
  assert type(val) is np.ndarray
  assert not check or np.all((val==0) + (val==1))
  return val*2-1

def scaleto01(val,check=True):
  if type(val) is not list and type(val) is not np.ndarray:
    val = [val]
  if type(val) is list:
    val = np.array(val)
  assert type(val) is np.ndarray
  assert not check or np.all((val==-1) + (val==1))
  return (val+1)/2.0


