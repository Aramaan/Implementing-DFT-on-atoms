import numpy as np
from matplotlib import pyplot as plt
import sys, os
sys.path.append(os.path.abspath("."))
from Packages.Integration import GaussQuad

d = 20*10**(3)
alpha = np.pi/d

def q(u):
    return (np.sin(alpha*u))**2

def I(x,f,wl,q):
   
    N = 100
    Integrand = lambda u:np.sqrt(q(u))*np.exp(1j*2*np.pi*x*u/(wl*f))
    I = GaussQuad(-w/2,w/2,N,Integrand) 
    return (np.abs(I))**2

wavl = 500
f = 1*(10)**(9)
slits = 10
w = d*slits
x = np.linspace(-5,5,1000)*10**(7)


Intensity = np.array([I(X,f,wavl,q) for X in x])
plt.plot(x*1e-7,Intensity*1e-10)
plt.grid()
plt.xlabel(r'$x (cm)$')
plt.ylabel(r'$I (\times 10^{10} Wm^{-2})$')
plt.savefig('Ques16/Q16(i)(a).png')
plt.show()

y = np.linspace(-0.5,0.5,10)*1e7
line = np.zeros((len(y),len(x)))

for i in range(len(x)):
    for j in range(len(y)):
        line[j,i] = Intensity[i]

plt.pcolor(x*1e-7,y*1e-7,line*1e-10,cmap='gray',vmax=0.25)
plt.colorbar()
plt.savefig('Ques16/Q16(i)(b).png')
plt.show()

##################################################################

q = lambda u: np.sin(alpha*u)**2*np.sin(alpha*u/2)**2

wavl = 500
f = 1*(10)**(9)
slits = 10
w = d*slits
x = np.linspace(-5,5,1000)*10**(7)


Intensity = np.array([I(X,f,wavl,q) for X in x])
plt.plot(x*1e-7,Intensity*1e-10)
plt.grid()
plt.xlabel(r'$x (cm)$')
plt.ylabel(r'$I (\times 10^{10} Wm^{-2})$')
plt.savefig('Ques16/Q16(ii)(a).png')
plt.show()

y = np.linspace(-0.5,0.5,10)*1e7
line = np.zeros((len(y),len(x)))

for i in range(len(x)):
    for j in range(len(y)):
        line[j,i] = Intensity[i]

plt.pcolor(x*1e-7,y*1e-7,line*1e-10,cmap='gray',vmax=0.25)
plt.colorbar()
plt.savefig('Ques16/Q16(ii)(b).png')
plt.show()

##################################################################
def I(x,f,wl,q):
    N = 100
    Integrand = lambda u:np.sqrt(q(u*1e-3))*np.exp(1j*2*np.pi*x*u/(wl*f))
    I = GaussQuad(-w/2,w/2,N,Integrand) 
    return (np.abs(I))**2

q = lambda u: np.piecewise(u, [np.logical_and(u>=-37.5, u<=-27.5), np.logical_and(u>=17.5, u<=37.5)], [1.,1.,0.])
    
w = 75*1e3
wavl = 500 
f = 1*1e9 
x = np.linspace(-5,5,1000)*1e7


Intensity = np.array([I(X,f,wavl,q) for X in x])
plt.plot(x*1e-7,Intensity*1e-10)
plt.grid()
plt.xlabel(r'$x (cm)$')
plt.ylabel(r'$I (\times 10^{10} Wm^{-2})$')
plt.savefig('Ques16/Q16(iii)(a).png')
plt.show()
y = np.linspace(-0.5,0.5,10)*1e7
line = np.zeros((len(y),len(x)))

for i in range(len(x)):
    for j in range(len(y)):
        line[j,i] = Intensity[i]

plt.figure(figsize=(13,1))
plt.pcolor(x*1e-7,y*1e-7,line*1e-10,cmap='gray')
plt.colorbar()
plt.savefig('Ques16/Q16(iii)(b).png')
plt.show()