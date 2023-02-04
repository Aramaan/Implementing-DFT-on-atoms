#%%
import numpy as np
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.differentiation import gradient
from matplotlib import pyplot as plt

h = np.loadtxt('Ques19/altitude.txt')
h = np.flipud(-h)

def Intensity(phi,h,dx,dy):
    phi = np.deg2rad(phi)
    Grad = gradient(h,dx,dy)
    return (np.cos(phi)*Grad[:,:,0]+ np.sin(phi)*Grad[:,:,1])/np.sqrt(Grad[:,:,0]**2+Grad[:,:,1]**2+1)

Int = Intensity(45,h,30000,30000)
plt.pcolormesh(Int,vmax=0.7*10**(-2),vmin=-0.7*10**(-2)) 
plt.savefig('Ques19/worldmap.png')
plt.show()
# Grad(h,30000,30000).sh

h = np.loadtxt('Ques19/stm.txt')
h = np.flipud(h)
Int = Intensity(45,h,2.5,2.5)
plt.pcolormesh(Int) 
plt.savefig('Ques19/silicon.png')
plt.show()



# %%
