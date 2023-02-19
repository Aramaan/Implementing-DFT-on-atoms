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

'''
The 1 th derivative of the function at z=0 is 2.0007389056098934 
The 2 th derivative of the function at z=0 is 4.001477811219787 
The 3 th derivative of the function at z=0 is 8.00443343365936 
The 4 th derivative of the function at z=0 is 16.017733734637435 
The 5 th derivative of the function at z=0 is 32.088668673187186 
The 6 th derivative of the function at z=0 is 64.53201203912306 
The 7 th derivative of the function at z=0 is 131.72408427386088 
The 8 th derivative of the function at z=0 is 285.79267419089143 
The 9 th derivative of the function at z=0 is 780.1340677180365 
The 10 th derivative of the function at z=0 is 3705.3406771800956 
The 11 th derivative of the function at z=0 is 31542.74744898395 
The 12 th derivative of the function at z=0 is 358032.9693878303 
The 13 th derivative of the function at z=0 is 4609372.602041123 
The 14 th derivative of the function at z=0 is 64432912.42856752 
The 15 th derivative of the function at z=0 is 966280694.4287686 
The 16 th derivative of the function at z=0 is 15460032358.8582 
The 17 th derivative of the function at z=0 is 262819567060.5842 
The 18 th derivative of the function at z=0 is 4730750109938.269 
The 19 th derivative of the function at z=0 is 89884247632394.02 
The 20 th derivative of the function at z=0 is 1797684943210963.2
'''