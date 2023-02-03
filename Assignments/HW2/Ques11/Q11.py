import numpy as np
from matplotlib import pyplot as plt
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages import Integration

#Integrands to be used in integration
Ic = lambda t: np.cos((1/2)*np.pi*t**2)
Is = lambda t: np.sin((1/2)*np.pi*t**2)

N = 50
#Computation of Building functions
#usng Gaussian Quadrature integration
C = lambda u: Integration.GaussQuad(0,u,N,Ic)
S = lambda u: Integration.GaussQuad(0,u,N,Is)

#Intensity in units of I0
I = lambda u: (1/8)*((2*C(u)+1)**2 + (2*S(u)+1)**2 )

x = np.linspace(-5,5,300)
wl = 1 #wavelength
z = 3 


Intensity = [I(u) for u in x*np.sqrt(2/(wl*z))]

plt.plot(x,Intensity)
plt.title(r'Diffraction of plane wave by a straight barrirer')
plt.xlabel(r'$x$')
plt.ylabel(r'$\frac{I}{I_0}$ at $(z=3)$')
plt.grid()
plt.savefig('Ques11/Q11.png')
plt.show()


