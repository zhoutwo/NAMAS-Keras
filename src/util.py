import random
import numpy as np


def string_shortfloat(t):
    return '%2.4g' % t

def shuffleTable(t):
    random.shuffle(t)

def string_split(s, c=' '):
   return s.split(c)

def add(tab, key):
   tab[key] = tab[key] or {}

def has(tab, key):
   return key in tab

def isnan(x):
    return np.isnan(x)
