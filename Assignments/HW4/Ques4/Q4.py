import numpy as np
import matplotlib.pyplot as plt

dow = np.loadtxt('Ques4/dow.txt')
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

'''
As we ignore the high frequency components, we basically remove the 
rapidly changing terms in the function and as a result the function
smoothens out
'''


N = 1000
t = np.linspace(0.,1,N)
dow = np.piecewise(t,[np.floor(2*t)%2 == 0,np.floor(2*t)%2 == 1],[1,-1])

plt.figure(figsize=(10,10))
plt.plot(t,dow)
plt.title('Dow Jones Indeustrial Average')
plt.xlabel('time in days')
plt.ylabel('Closing values')
plt.grid()
plt.savefig('Ques4/4(b)(i).png')
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
plt.savefig('Ques4/4(b)(ii).png')
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
plt.savefig('Ques4/4(b)(iii).png')
plt.show()

'''
The reason for the wiggles is that the function given to us is not sinosuidal and 
is constant and undurgoes abrupt changes. So it has got both low frequency componenets and 
high frequency components and when we remove thee high frequency components, it is difficult to
model the abrupt change with these low frequency components and that leads to those wiggles and
resulting imperfection in digital compression
'''