import numpy as np
from matplotlib import pyplot as plt

# N is power of 2
import numpy as np

def fft(x):
    n = len(x)
    if n == 1:
        return x
    else:
        even = fft(x[::2])
        odd = fft(x[1::2])
        t = np.exp(-2j * np.pi * np.arange(n) / n)
        return np.concatenate([even + t[:n//2] * odd, even + t[n//2:] * odd])
    
df = np.loadtxt('Ques6/pitch.txt')

plt.figure(figsize=(10,10))
plt.plot(df)
plt.title(r'pitch')

plt.grid()
plt.savefig('Ques2/Q2(i).png')
plt.show()

fft_spots = np.abs(np.fft.fft(df))**2

plt.figure(figsize=(10,10))
plt.plot(fft_spots)
plt.title(r'FFT pitch')

plt.xlim(0,300)
plt.grid()  
plt.savefig('Ques2/Q2(ii).png')
plt.show()

fft_spots = np.abs(fft(df))**2

plt.figure(figsize=(10,10))
plt.plot(fft_spots)
plt.title(r'User definied FFT pitch')
plt.xlim(0,300)
plt.grid()  
plt.savefig('Ques2/Q2(ii).png')
plt.show()
