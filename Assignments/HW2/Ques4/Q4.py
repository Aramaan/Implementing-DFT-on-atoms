
from matplotlib import pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.abspath("."))
from Packages.Integration import simpson

def I(theta,m,x):
    j = np.cos(m*theta-x*np.sin(theta))/np.pi   
    return j

def J(m,x):
    A = lambda theta: I(theta,m,x)
    In = simpson(0,np.pi,1000,A)
    return In

x = np.linspace(0,20,100)
for m in range(3):
    j = [J(m,k) for k in x ]
    plt.plot(x,j)
plt.legend(['$J_{}$'.format(i) for i in range(3)])
plt.title('Bessel Function')
plt.savefig('Ques4/Q4(i).png')
plt.show()

def Intensity(r,wl):
    k = 2*np.pi/wl
    I = (J(1,k*r)/k*r)**2
    return I

x = np.linspace(-1e-6,1e-6,100)
y = np.linspace(-1e-6,1e-6,100)
xm, ym =  np.meshgrid(x,y)
wl = 500e-9
Int = lambda x,y : Intensity(np.sqrt(x**2+y**2),wl)
Data = np.zeros((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        Data[i,j] = Int(x[i],y[j])
plt.pcolormesh(x,y,Data)
plt.colorbar()
plt.xlabel('in micrometers')
plt.ylabel('in micrometers')
plt.title('Diffraction Pattern')
plt.savefig('Ques4/Q4(ii).png')
plt.show()


