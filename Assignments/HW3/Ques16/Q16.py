
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import electron_volt,elementary_charge,electron_mass,hbar

'''
16(b)
'''
a = 10*electron_volt
L = 5e-10

def V(x):
    return a*x/L

def H(m,n):
    if m==n:
        kin = (-hbar**2/(2*electron_mass))*(-(n*np.pi/L)**2)
        pot = (2*a/L**2)*((L**2)/4)
    elif((m+n)%2 == 1):
        kin = 0
        pot = ((-(2*a/L**2)*(2*L/np.pi)**2*m*n/(m**2-n**2)**2))
    else:
        kin = 0
        pot = 0
    return (kin +pot)/electron_volt

Hamiltonian1 = np.zeros([10,10])
for i in range(10):
    for j in range(10):
        Hamiltonian1[i][j] = H(i+1,j+1)

eig = np.sort(np.linalg.eigvals(Hamiltonian1))
print(eig)

Hamiltonian2 = np.zeros([100,100])
for i in range(100):
    for j in range(100):
        Hamiltonian2[i][j] = H(i+1,j+1)

eig = np.sort(np.linalg.eigvals(Hamiltonian2))
print(eig[0:10])

phi = np.zeros([10,1])
l,v = np.linalg.eig(Hamiltonian1)
print("djfelllllllllllllllllllll")
print(l,v)

def wavef(x,v):
    sum = 0
    for i in range(10):
        sum += np.absolute(v[i]*np.sin(np.pi*(i+1)*x/L))
    return sum

x = np.linspace(0,L,200)
plt.plot(x,wavef(x,v[:,0]))
plt.show()
plt.plot(x,wavef(x,v[:,1]))
plt.show()
plt.plot(x,wavef(x,v[:,2]))
plt.show()

