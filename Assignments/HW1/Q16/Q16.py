#%%
import numpy as np
from matplotlib import pyplot as plt

# %%
r = np.arange(0,4,0.01)
def map(x,r):
    return r*x*(1-x)
def iterate(r):
    x = 0.5
    for i in range(1000):
        x = map(x,r)
    return x
def iterate2(r):
    x = []
    x.append(iterate(r))
    for i in range(1,1000):
        x.append(map(x[i-1],r))
    return x



# %%
for i in r:
    l = iterate2(i)
    for j in l: 
        plt.plot(i,j,'k.')
plt.savefig('feigenbaum_plot.png')
plt.show()
# %%
