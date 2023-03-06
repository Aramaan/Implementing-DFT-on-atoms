import numpy as np
import matplotlib.pyplot as plt

df = np.loadtxt('Ques2/sunspots.txt')
time = df[:,0]
nspots = df[:,1]

plt.figure(figsize=(10,10))
plt.plot(time,nspots)
plt.title(r'Sunspots by month')
plt.xlabel(r'time in Months')
plt.ylabel(r'Number of Sunspots')
plt.grid()
plt.savefig('Ques2/Q2(i).png')
plt.show()

N = len(time)
s = time[1] - time[0]
f = (1/np.linspace(0,1/s,N))
fft_spots = np.abs(np.fft.fft(nspots))**2

plt.figure(figsize=(10,10))
plt.plot(f,fft_spots)
plt.title(r'Sunspots by month')
plt.xlabel(r'time period in months')
plt.ylabel(r'Fourier transformed #sunspots')
plt.xlim(0,300)
plt.grid()  
plt.savefig('Ques2/Q2(ii).png')
plt.show()

print('The Dominant period of sunspots is  %.3f months'%f[np.argmax(fft_spots[1:])])
#taking from 1 so as to avoid the 0 peak

"""
The Dominant period of sunspots is 136.6 months
"""