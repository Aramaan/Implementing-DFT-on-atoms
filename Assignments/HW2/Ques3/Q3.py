
#%%
from matplotlib import pyplot as plt
import numpy as np
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Ques2.Q2 import simpson

def f(t):
    return np.exp(-t**2)

x = np.arange(0,3,0.01)
E = [simpson(0,x[i],i+1,f) for i in range(len(x))]
plt.plot(x,E)
plt.show()


# %%
