import numpy as np
import matplotlib.pyplot as plt

d = 20 #micrometers
alpha = np.pi/d
w = 200 #micrometers
factor = 10
W = factor*w
wavelength = 0.5 
f = 1*1e6 #micrometers
N = 1000
NN = N*factor #no of points after padding
screen = 1e5 #micrometers

def qq(u):
    return np.sin(alpha*u)**2
u = np.linspace(-w/2,w/2,N)
q = qq(u)

Q  = np.zeros([NN,])
print(Q)
for i in range((NN-N)//2,(NN+N)//2):
    Q[i] = q[i-(NN-N)//2]
U = np.linspace(-W/2,W/2,NN)
y = np.sqrt(Q)

plt.plot(U,Q)
plt.savefig('Ques7/Q7(transmission).png')
plt.show()

xlim = (NN-1)*wavelength*f/W
x = np.linspace(0,xlim,NN)

I = (W/NN)**2*np.abs(np.fft.fft(y))**2
limit = int(screen/(2*wavelength*f/W))
x = x[:limit]
I = I[:limit]

x = np.hstack((np.flip(-x),x))
I = np.hstack((np.flip(I),I))

plt.figure(figsize=(10,10))
plt.plot(x*1e-4,I*1e-4)
plt.title('Diffraction Pattern')
plt.ylabel('I in 10^(4) W m^(-2)')
plt.xlabel('x in cm')
plt.grid()
plt.savefig('Ques7/Q7.png')
plt.show()

