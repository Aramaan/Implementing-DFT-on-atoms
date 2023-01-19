#%%
import numpy as np
from matplotlib import pyplot as plt

# %%
r = np.arange(0,4,0.01)

'''
Performs the map operation once on 
the input value x wih a given r  for rate

input;
x = input to map
r = rate constant of map

output;
ouput of map
'''
def map(x,r):
    return r*x*(1-x)

'''
Repeats the map operation with initial input 0.5
1000 times for the output values to reach
 equilibrium (cycle or point)

input;
r = rate constant of map function

output;
value of the map after thousandth iteration
'''
def iterate(r):
    x = 0.5
    for i in range(1000):
        x = map(x,r)
    return x

'''
Takes the output from iterate() and 
stores the next thousand values in an array x
 and returns it as output

input;
r = rate constant of map function

output;
array of 1000 elements, values of output of map after equilibrium
'''
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
