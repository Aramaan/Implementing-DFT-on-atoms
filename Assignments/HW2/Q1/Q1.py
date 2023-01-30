#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('velocities.txt',delimiter='\t',names=['time','velocities'])
t = df['time']
v = df['velocities']

def trap(a,b,c,d):
    area =((b-a)/2)*(d+c)
    return area

def tIntegraion(x,y):
    integral = [0]
    for i in range(1,len(x)):
        area = trap(x[i-1],x[i],y[i-1],y[i])
        integral.append( integral[i-1] + area)
    return integral

df
# %%
plt.plot(t,v)
x = tIntegraion(t,v)
x
lt.plot(t,x)
# %%
