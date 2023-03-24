import numpy as np
import matplotlib.pyplot as plt


N = int(1e5)

u1 = np.random.uniform(0,2*np.pi,size=N)

u2 = np.random.uniform(0,1,size=N)
x2 = np.piecewise(u2,[u2<=0.5],[lambda u: np.arccos(1-2*u),lambda u: np.pi-np.arccos(2*u-1)])


phi = np.linspace(0, 2*np.pi, N)
theta = np.linspace(0, np.pi, N)

plt.figure(figsize=(10,10))
plt.title(r'Azimuthal Angle')
plt.hist(u1,bins=25,color='yellow')
plt.ylabel(r'P(phi)')
plt.xlabel(r'Angle')
plt.grid()

plt.savefig('Ques10/10(i).png')
plt.show()

plt.figure(figsize=(10,10))
plt.hist(x2,bins=25,color='yellow')
plt.title(r'Polar Angle')
plt.ylabel(r'P(theta)')
plt.xlabel(r'Angle')
plt.grid()
plt.savefig('Ques10/10(ii).png')
plt.show()

'''
P(theta)dtheta = sin(theta)dtheta
P(phi)dphi = dphi

Polar angle B is from 0 to pi
Azimuthal angle 0 is from 0 to 2pi

Integrating, both of them reduce to 1 and hence normalised

P(phi) can be sampled from a uniform distribution with support from 0 to 2i, 
However, P(theta) needs an inverse sampling from a uniform distribution from 0 to 1 

The Cumulative Distribution for P(theta) is 
C(theta)= P(theta')dtheta' = sin (theta')dtheta' = (1 - cos (theta))/2
Let u be sampled from a uniform distribution with support from 0 to 1 Obviously,
C(C-'u))= u 

Therefore substituting gives the following 
invC(u) transformation that
samples from P(theta) =  arcos(1 - 2u) u <= 1/2
                    pi - arcos(2u - 1) u > 1/2

'''