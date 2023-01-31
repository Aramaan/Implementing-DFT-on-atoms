
#%%
from matplotlib import pyplot as plt
import numpy as np

def f(t):
    return np.exp(-t**2)

def step(a,b,s,f):
    x = np.arange(a,b,s)
    y = f(x)
    return x,y

from Q2.Q2 import sIntegration

x,y = step(0,3,0.1,f)
I = sIntegration(x,y)
plt.plot(x[::2],I)
plt.show()
 

# %%
