
#%%
from matplotlib import pyplot as plt
import numpy as np
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Ques2.Q2 import sIntegration

def f(t):
    return np.exp(-t**2)

def step(a,b,s,f):
    x = np.arange(a,b,s)
    y = f(x)
    return x,y



x,y = step(0,3,0.1,f)
plt.plot(x,y)
I = sIntegration(x,y)
plt.plot(x[::2],I)
plt.show()
# print(os.path.abspath('.'))
# print(Path('Q2.py').resolve())

# %%
