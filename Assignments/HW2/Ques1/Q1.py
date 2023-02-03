#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.Integration import tIntegraion

df = pd.read_csv('velocities.txt',delimiter='\t',names=['time','velocities'])
t = df['time']
v = df['velocities']

df
# %%
plt.plot(t,v)
x = tIntegraion(t,v)
x
plt.plot(t,x)
# %%
