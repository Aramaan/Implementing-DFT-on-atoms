import numpy as np
from matplotlib import pyplot as plt
import sys, os
from scipy.constants import epsilon_0
sys.path.append(os.path.abspath("."))
from Packages.Integration import GaussQuad2d
from Packages.differentiation import gradient


e0 = epsilon_0
pi = np.pi


def phi(x,y,q,x0,y0):
    return (q/(4*pi*e0*np.sqrt((x-x0)**2+(y-y0)**2)))

x = np.arange(-0.5,0.5,0.01)
y = np.arange(-0.5,0.5,0.01)
xm, ym = np.meshgrid(x,y)

phi1 = phi(xm,ym,-1,-0.05,0)
phi2 = phi(xm,ym,1,0.05,0)
total = (phi1+phi2)*1e-10

plt.pcolormesh(100*x,100*y,total,vmax = 1, vmin = -1) 
#x,y in cm 
#potential in units of 0.1nV
plt.colorbar()

plt.xlabel('x(cm)')
plt.ylabel('y(cm)')
plt.title(r'Potential in units of 0.1nV')

plt.scatter([-5],[0],c='b',marker='o',s=100)
plt.scatter([-5],[0],c='w',marker='_',s=50)
plt.scatter([5],[0],c='r',marker='o',s=100)
plt.scatter([5],[0],c='w',marker='+',s=50)
plt.savefig('Ques17/Q17(i).png')
plt.show()

grad = gradient(total,0.01,0.01)
Ex = -grad[:,:,1]
Ey = -grad[:,:,0]
# Ex = np.piecewise(Ex,[Ex>=1,Ex<=-1],[1,-1,lambda Ex: Ex])
# Ey = np.piecewise(Ey,[Ey>=1,Ey<=-1],[1,-1,lambda Ey: Ey])


plt.streamplot(x*100,y*100,Ex,Ey,color='white')
plt.pcolormesh(x*100,y*100,np.sqrt(Ex**2+Ey**2),vmax=10,vmin=0)    
plt.colorbar()

plt.xlabel('x(cm)')
plt.ylabel('y(cm)')
plt.title(r'Electric field in units of 0.1nV/m')
plt.scatter([-5],[0],c='b',marker='o',s=100)
plt.scatter([-5],[0],c='w',marker='_',s=50)
plt.scatter([5],[0],c='r',marker='o',s=100)
plt.scatter([5],[0],c='w',marker='+',s=50)
plt.savefig('Ques17/Q17(ii).png')
plt.show()
plt.show()

L = 0.1
k = 2*np.pi/L
sigma = lambda x,y : 100*np.sin(k*x)*np.sin(k*y)

def phi(x,y,sigma):
    Integrand = lambda x0,y0: sigma(x0,y0)/(4*pi*e0*np.sqrt((x-x0)**2+(y-y0)**2))
    Integral = GaussQuad2d(-L/2,L/2,99,Integrand)
    return Integral

# def phi(x,y,sigma):
#     I = lambda xp,yp:sigma(xp,yp)/np.sqrt((x-xp)**2+(y-yp)**2)
#     return (1./(4*np.pi*e0))*GaussQuad2d(I,[-L/2.,-L/2.],[L/2.,L/2.],[N,N])



total = np.zeros((len(y),len(x)))
for i in range(len(x)):
    for j in range(len(y)):
        total[j,i] = phi(x[i],y[j],sigma)*1e-9

plt.pcolormesh(100*x,100*y,total,vmax = 0.5, vmin = -0.5) 
#x,y in cm 
#potential in units of 0.1nV
plt.colorbar()

plt.xlabel('x(cm)')
plt.ylabel('y(cm)')
plt.title(r'Potential in units of nV')

plt.scatter([-5],[0],c='b',marker='o',s=100)
plt.scatter([-5],[0],c='w',marker='_',s=50)
plt.scatter([5],[0],c='r',marker='o',s=100)
plt.scatter([5],[0],c='w',marker='+',s=50)
plt.savefig('Ques17/Q17(iii).png')
plt.show()

grad = gradient(total,0.01,0.01)
Ex = -grad[:,:,1]
Ey = -grad[:,:,0]
# Ex = np.piecewise(Ex,[Ex>=1,Ex<=-1],[1,-1,lambda Ex: Ex])
# Ey = np.piecewise(Ey,[Ey>=1,Ey<=-1],[1,-1,lambda Ey: Ey])


plt.streamplot(x*100,y*100,Ex,Ey,color='white')
plt.pcolormesh(x*100,y*100,np.sqrt(Ex**2+Ey**2),vmax=10,vmin=0)    
plt.colorbar()

plt.xlabel('x(cm)')
plt.ylabel('y(cm)')
plt.title(r'Electric field in units of nV/m')
plt.scatter([-5],[0],c='b',marker='o',s=100)
plt.scatter([-5],[0],c='w',marker='_',s=50)
plt.scatter([5],[0],c='r',marker='o',s=100)
plt.scatter([5],[0],c='w',marker='+',s=50)
plt.savefig('Ques17/Q17(iv).png')
plt.show()
plt.show()