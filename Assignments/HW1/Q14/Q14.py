#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
'''
14(a)
'''
#%%
t = np.linspace(0,2*np.pi)
x = 2*np.cos(t) + np.cos(2*t)
y = 2*np.sin(t) - np.sin(2*t)
plt.plot(x,y)

'''
14(b)
'''
# %%
t = np.linspace(0,10*np.pi,1000)
x = t**2*np.cos(t)
y = t**2*np.sin(t)
plt.plot(x,y)
'''
14(c)
'''
# %%
t = np.linspace(0,24*np.pi,10000)
x = (np.exp(np.cos(t))-2*np.cos(4*t)+(np.sin(t/12))**5)*np.cos(t)
y = (np.exp(np.cos(t))-2*np.cos(4*t)+(np.sin(t/12))**5)*np.sin(t)
plt.plot(x,y)

# %%
