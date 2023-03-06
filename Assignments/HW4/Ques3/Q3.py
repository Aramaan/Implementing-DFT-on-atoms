import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

piano = np.loadtxt('Ques3/piano.txt')
trumpet = np.loadtxt('Ques3/trumpet.txt')

N = len(piano)
freq = 44100
period = 1/freq
t = np.linspace(0,N*period,N)

plt.figure(figsize=(10,10))
plt.plot(t,piano)
plt.title('Piano Music')
plt.xlabel('time')
plt.ylabel('function')
plt.grid()
plt.savefig('Piano.png')
plt.show()


plt.figure(figsize=(10,10))
plt.plot(t,trumpet)
plt.title('Piano Music')
plt.xlabel('time')
plt.ylabel('function')
plt.grid()
plt.savefig('Piano.png')
plt.show()

FTP = (np.abs(np.fft.fft(piano))**2)[:N//2]
FTT = (np.abs(np.fft.fft(trumpet))**2)[:N//2]

f = np.linspace(0,freq//2,N//2)
s= 200//(f[1]-f[0])

peaks = signal.find_peaks(FTP,distance=s,threshold=1e13)[0]
print('Frquencies of piano ')
p = [f[i] for i in peaks]
print(p)

plt.figure(figsize=(10,10))
plt.plot(f,FTP)
plt.title('fourier transform piano')
plt.xlabel('f (Hz)')
plt.ylabel('amplitude')
plt.grid()
plt.savefig('Piano(ii).png')
plt.show()


peaks = signal.find_peaks(FTT,distance=s,threshold=1e13)[0]
print('frquencies of trumpet ')
p = [f[i] for i in peaks]
print(p)

plt.figure(figsize=(10,10))
plt.plot(f,FTT)
plt.title('Fourier transform trunmpet')
plt.xlabel('f (Hz)')
plt.ylabel('amplitude')
plt.grid()
plt.savefig('Trumpet(ii).png')
plt.show()


"""
Output:
    
Frquencies of piano
[ 524.79524795 1051.35451355 1578.35478355]
frquencies of trumpet
[ 521.70821708 1043.85743857 1566.00666007 2087.71487715 2609.86409864
 3132.01332013 3653.72153722 4175.87075871]
The number of higher harmonics is more for trumpets
"""