import numpy as np
from matplotlib import pyplot as plt
from math import factorial
import sys, os
sys.path.append(os.path.abspath("."))
from Packages.Integration import GaussQuad

'''
Quantum uncertainity in the harmonic oscillator
'''

'''
Hermite functions
'''
def H(n,x):
    if n < 0:
        raise ValueError
    elif n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        return 2*x*H(n-1,x) - 2*n*H(n-2,x)
    
x = np.linspace(-4,4,100)

'''
Harmonic Oscillator Wavefunctions
'''
def phi(n,x):
    p = H(n,x)*np.exp(-(x**2)/2)/(np.sqrt(2**n*factorial(n)*np.sqrt(np.pi)))
    return p

for n in range(4):
    plt.plot(x,phi(n,x))
plt.legend(range(4))
plt.title('First Few Harmonic Oscillator Wavefunctions')
plt.xlabel('position')
plt.ylabel('wavefunction')
plt.savefig('Ques13/Q13(i)')
plt.show()


y = np.linspace(-10,10,500)             
plt.plot(y,phi(30,y))
plt.title('Harmonic Oscillator Wavefunctions for n=30')
plt.xlabel('position')
plt.ylabel('wavefunction')
plt.savefig('Ques13/Q13(ii)')
plt.show()

'''
Uncertainity
'''
def RootSquareUncertainity(n):
    f = lambda x: x**2*np.abs(phi(n,x))**2
    u = GaussQuad(-10,10,100,f)
    return u
print('The uncertainity or the root mean square position of the particle is {} for n = 5'.format(RootSquareUncertainity(5)))