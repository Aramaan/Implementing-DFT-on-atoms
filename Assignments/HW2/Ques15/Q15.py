import numpy as np
from matplotlib import pyplot as plt
import sys, os
sys.path.append(os.path.abspath("."))
from Packages.Integration import GaussQuad

'''
Gamma Function
'''

def Integrand(a,x):
    return x**(a-1)*np.exp(-x)

x = np.linspace(0,5,100)


for a in range(2,5):
    y = Integrand(a,x)
    plt.plot(x,y)
plt.show()

def Int(a,x):
    return np.exp((a-1)*np.log(x)-x)

def Int2(a,z):
    c = a-1
    return Int(a,c/(1/z-1))*c/(1-z)**2


def gamma(a):
    I =lambda z: Int2(a,z)
    return GaussQuad(0,1,100,I)

print('Gamma(3/2) is {}'.format(gamma(3/2)))

for i in range(10):
    print('Gamma({}) is {}'.format(i,gamma(i)))
    

