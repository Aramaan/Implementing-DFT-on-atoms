import numpy as np
from math import factorial
'''
Differenciatin by integrating
'''

f = lambda z : np.exp(2*z)
N = 10000

z = lambda k: np.exp(1j*2*np.pi*k/N)

def deriv(func,ord,N):
    k = np.arange(0,N+1,1)
    #print(k)
    zk = z(k)
    derivative = (factorial(ord)/N)*np.sum(func(zk)*np.exp(-1j*2*np.pi*k*ord/N))
    return derivative

for i in range(1,21):
    print("The {} th derivative of the function at z=0 is {} ".format(i,np.real(deriv(f,i,N))))

