import numpy as np
import matplotlib.pyplot as plt

dow = np.loadtxt('Ques5/dow2.txt')
N = len(dow)
t = np.arange(0,N,1)#in days
#representing dates on a graph would be difficult

plt.figure(figsize=(10,10))
plt.plot(t,dow)
plt.title('Dow Jones Indeustrial Average')
plt.xlabel('time in days')
plt.ylabel('Closing values')
plt.grid()
plt.savefig('Ques4/4(a)(i).png')
plt.show()
s = t[1]-t[0]
f = np.linspace(0,1/s,N//2+1)
fft_dow = np.fft.rfft(dow)

plt.figure(figsize=(10,10))
plt.semilogy(f,(np.abs(fft_dow))**2)
plt.title('Power Spectrum of Closing values')
plt.xlabel('frequency in 1/day')
plt.ylabel('Fourier transformed closing values')
plt.grid()
plt.savefig('Ques4/4(a)(ii).png')
plt.show()

def smoothen(fourier,perc):
    f = fourier.copy()
    N = len(fourier)
    for i in range(N):
        if (i/N >= perc/100):
            f[i] = 0
    return f

fft_dow_10 = smoothen(fft_dow,10)
fft_dow_2 = smoothen(fft_dow,2)

dow_10 = np.fft.irfft(fft_dow_10)
dow_2 = np.fft.irfft(fft_dow_2)
plt.figure(figsize=(10,10))
plt.plot(t,dow)
plt.plot(t,dow_10,label='10 percent')
plt.plot(t,dow_2,label='2 percent')
plt.title('Dow Jones Industrial Average with varying smotthness as we reject higher freq components')
plt.xlabel('time in days')
plt.ylabel('Closing values')
plt.legend(loc='best')
plt.savefig('Ques4/4(a)(iii).png')
plt.show()

def dct(y):
    N = len(y)
    y2 = np.empty(2*N,float)
    y2[:N] = y[:]
    y2[N:] = y[::-1]

    c = np.fft.rfft(y2)
    phi = np.exp(-1j*np.pi*np.arange(N)/(2*N))
    return np.real(phi*c[:N])

def idct(a):
    N = len(a)
    c = np.empty(N+1,complex)

    phi = np.exp(1j*np.pi*np.arange(N)/(2*N))
    c[:N] = phi*a
    c[N] = 0.0
    return np.fft.irfft(c)[:N]

plt.figure(figsize=(10,10))
plt.plot(t,dow)
plt.title('Dow Jones Indeustrial Average')
plt.xlabel('time in days')
plt.ylabel('Closing values')
plt.grid()
plt.savefig('Ques4/4(a)(i).png')
plt.show()
s = t[1]-t[0]
f = np.linspace(0,1/s,N//2+1)
fft_dow = dct(dow)

def smoothen(fourier,perc):
    f = fourier.copy()
    N = len(fourier)
    for i in range(N):
        if (i/N >= perc/100):
            f[i] = 0
    return f

fft_dow_10 = smoothen(fft_dow,10)
fft_dow_2 = smoothen(fft_dow,2)

dow_10 = idct(fft_dow_10)
dow_2 = idct(fft_dow_2)
plt.figure(figsize=(10,10))
plt.plot(t,dow)
plt.plot(t,dow_10,label='10 percent')
plt.plot(t,dow_2,label='2 percent')
plt.title('Dow Jones Industrial Average with varying smotthness as we reject higher freq components')
plt.xlabel('time in days')
plt.ylabel('Closing values')
plt.legend(loc='best')
plt.savefig('Ques4/4(a)(iii).png')
plt.show()
