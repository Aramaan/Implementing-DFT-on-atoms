
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.Integration import tIntegraion

df = pd.read_csv('Ques1/velocities.txt',delimiter='\t',names=['time','velocities'])
t = df['time']
v = df['velocities']

plt.plot(t,v)
plt.xlabel('time')
x = tIntegraion(t,v)
plt.plot(t,x)
plt.legend(['velocity','position'])
plt.show()

