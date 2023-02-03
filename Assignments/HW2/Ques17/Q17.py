import numpy as np
from matplotlib import pyplot as plt
import sys, os
from scipy.constants import epsilon_0
sys.path.append(os.path.abspath("."))
from Packages.Integration import GaussQuad

e0 = epsilon_0
pi = np.pi

def phi(x,y,q,x0,y0):
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    return (q/(4*pi*e0*r))

x = np.arange(-0.5,0.5,0.01)
y = np.arange(-0.5,0.5,0.01)
xm, ym = np.meshgrid(x,y)

phi1 = phi(xm,ym,1,0.05,0)
phi2 = phi(xm,ym,-1,-0.05,0)
total = phi1 + phi2

plt.pcolormesh(100*x,100*y, total)
plt.colorbar()