import numpy as np

#phs=placeholders
#curs=current values
def make_feed_dict(phs, curs):
  assert len(phs) == len(curs)
  def quant(x):
    if (x >0.8):
      return 1
    elif (x < -0.8):
      return -1
    else:
      return 0
  quant = np.vectorize(quant,otypes=[np.float])
  news = [quant(cur) for cur in curs]
  print news
  feed_dict = {}
  for i in range(len(phs)):
    feed_dict[phs[i]] = news[i]
  return feed_dict

def bitstr(i,N):
  ret = bin(i)[2:][::-1]
  for _ in range(N-len(ret)):
    ret = ret+'0'
  return ret

def bitfield(i,N):
  bits = bitstr(i,N)
  bits = [int(digit)*2-1 for digit in bits]
  return np.array(bits).astype(float)

def scaleto11(val,check=True):
  if type(val) is int:
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


