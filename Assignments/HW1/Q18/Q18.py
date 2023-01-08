#%%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# %%
data = pd.read_csv('Q18/millikan.txt',delimiter=' ')

# %%
def Average(a,N):
    return np.sum(a)/N

def crAverage(a,b,N):
    return np.sum(np.dot(a,b))/N

# %%
def line(x,y):
    N = len(x)
    m = (crAverage(x,y,N)-Average(x,N)*Average(y,N))/(crAverage(x,x,N)-Average(x,N)*Average(x,N))
    c = (crAverage(x,x,N)*Average(y,N)-Average(x,N)*crAverage(x,y,N))/(crAverage(x,x,N)-Average(x,N)*Average(x,N))
    return [m,c]
# %%
x = data.iloc[:,0]
y = data.iloc[:,1]
plt.scatter(x,y)
l = line(x,y)
plt.plot(x,l[0]*x + l[1],'g-.')
plt.title('Number of Sunspots vs Time')
plt.xlabel('Time in months')
plt.ylabel('Number of Sunspots')
plt.savefig("Q18/Q18_a.png")
plt.show()
# %%
