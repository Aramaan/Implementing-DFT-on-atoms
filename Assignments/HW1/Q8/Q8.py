#%%
import numpy as np
from scipy.constants import epsilon_0,elementary_charge
'''
The Madelung Constant
'''
#%%
C = elementary_charge/(4*np.pi*epsilon_0*1)

def V(i,j,k):
    return (-1)**(i+j+k)*1/(1*np.sqrt(i**2 + j**2 + k**2))     

def Vtotal(L):
    sum = 0
    for i in range(-L,L+1):
        for j in range(-L,L+1):
            for k in range(-L,L+1):
                if (i == 0) and (j == 0) and (k == 0):
                    continue
                sum = sum + V(i,j,k)
    return sum

print(Vtotal(100))
# %%
