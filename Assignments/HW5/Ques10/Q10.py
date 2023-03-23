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