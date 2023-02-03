import numpy as np
from matplotlib import pyplot as plt
import sys, os
sys.path.append(os.path.abspath("."))
from Packages.Integration import GaussQuad

d = 20*10**(3)
alpha = np.pi/d

def q(u):
    return (np.sin(alpha*u))**2

def I(x,d,slits,f,wl,q):
    w = d*slits
    N = 100
    Integrand = lambda u:np.sqrt(q(u))*np.exp(1j*2*np.pi*x*u/(wl*f))
    #u = np.linspace(-w/2,w/2,1000)
    #plt.plot(u,real(u))
    #plt.plot(u,imag(u))
    #plt.grid()
    #plt.show()
    I = GaussQuad(-w/2,w/2,N,Integrand) 
    return (np.abs(I))**2

wavl = 500
f = 1*(10)**(9)
slits = 10
x = np.linspace(-5,5,1000)*10**(7)


Intensity = np.array([I(X,d,slits,f,wavl,q) for X in x])
plt.plot(x*1e-7,Intensity*1e-10)
plt.grid()
plt.xlabel(r'$x (cm)$')
plt.ylabel(r'$I (\times 10^{10} Wm^{-2})$')
plt.savefig('16.png')
plt.show()

y = np.linspace(-0.5,0.5,10)*1e7
line = np.zeros((len(y),len(x)))

for i in range(len(x)):
    for j in range(len(y)):
        line[j,i] = Intensity[i]

plt.pcolor(x*1e-7,y*1e-7,line*1e-10,cmap='gray',vmax=0.25)
plt.colorbar()
plt.savefig('16_1_2.png')
plt.show()

q = lambda u,alpha: np.sin(alpha*u)**2*np.sin(alpha*u/2)**2