import numpy as np
import math



def create_layers(first,last,steps):
  r = (last/first)**(1/steps)
  return [int(first*(r**i) + 0.5) for i in range(steps+1)]
  

def histogram(Ws):
  print (len(Ws))
  hist = np.array([])
  for W in Ws:
    W_flatten = W.flatten()
    hist = np.append(hist,W.flatten())
  print (hist.shape)
  return hist


def depbits(val):
  bits = [0 for i in range(4)]
  bits[0] = 0b0101010101010101
  bits[1] = 0b0011001100110011
  bits[2] = 0b0000111100001111
  bits[3] = 0b0000000011111111
  
  tot = sum([ (val&(bits[i])) != ((val& (~(bits[i])))>>2**i) for i in range(4)])
  return tot

def histLut(Ws):
  hist = np.zeros(2**16)
  mult = [2**i for i in range(16)]
  for W in Ws:
    bW = W > 0
    for i in range(W.shape[0]):
      idx = sum(bW[i]*mult)
      hist[idx] = hist[idx] + 1
  return hist

def histLutDeps(Ws):
  hist = np.zeros(5)
  mult = [2**i for i in range(16)]
  for W in Ws:
    bW = W > 0
    for i in range(W.shape[0]):
      idx = sum(bW[i]*mult)
      numbits = depbits(idx)
      hist[numbits] = hist[numbits] + 1
  return hist

def log(b):
  def f(x):
    return math.log(x)/math.log(b)
  return f

def flatten_list(l):
  return [item for sublist in l for item in sublist]

#phs=placeholders
#curs=current values
def make_feed_dict(phs, curs,thresh=0,keep=False,prob=1):
  assert len(phs) == len(curs)
  def quant(x):
    if (x >=thresh and np.random.uniform() < prob):
      return 1
    elif (x < -thresh and np.random.uniform() < prob):
      return -1
    else:
      if keep:
        return x
      else:
        return np.random.normal(0,0.25)
  quant = np.vectorize(quant,otypes=[np.float])
  news = [quant(cur) for cur in curs]
  feed_dict = {}
  for i in range(len(phs)):
    feed_dict[phs[i]] = news[i]
  return feed_dict

def make_feed_dict2(phs, curs, ri=0):
  assert len(phs) == len(curs)
  def quant(x):
    if (x >=1):
      return 1
    elif (x < -1):
      return -1
    elif (np.random.random() < ri):
      if (x > (1-ri)):
        return 1
      elif (x < (ri-1)):
        return -1
      else:
        return x
    else:
      if ((np.random.random()*2 - 1.0) < x):
        return 1
      else:
        return -1
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


