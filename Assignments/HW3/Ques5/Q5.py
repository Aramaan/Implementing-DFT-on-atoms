import numpy as np
from scipy.constants import h,c,k
from matplotlib import pyplot as plt

def I(wl,T):
    wl *= 1e-9 #in nm
    #T *= 1e3   #in 10^3 K
    b = h*c/(wl*k*T)
    a = 2*np.pi*h*c**2
    return (a*wl**(-5))/(np.exp(b)-1)

def bisection(f,a,b,error):
    c = 0.
    while(np.abs((b-a)/2)>error):
        c = (a+b)/2
        if (f(c)*f(a)>=0): a = c
        else: b = c
    return (a+b)/2

def dfdt(x,f,d):
    return (f(x+d)-f(x-d))/(2*d)
T = np.linspace(4000,10000,100)
W = np.zeros(len(T))

for t in range(len(T)):
    In = lambda y: I(y,T[t])
    dIdx = lambda x: dfdt(x,In,1e-2)
    W[t] = (bisection(dIdx,10,10000,1e-6))

plt.plot(1/(T),W)
plt.xlabel(r'$\frac{1}{T [K]}$')
plt.ylabel(r'$\lambda_{max} [nm]$')
plt.show()
Wein = np.average(W*1e-9*T)
print("THe Wein displacement constant is the slope of the graph which is {} mK".format(Wein))
print("The temperature of the sun is {} K".format(Wein/(502*1e-9)))

'''
THe Wein displacement constant is the slope of the graph which is 0.0028977719567172696 mK
The temperature of the sun is 5772.454097046353 K
'''