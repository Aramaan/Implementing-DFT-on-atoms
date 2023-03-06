import numpy as np
import matplotlib.pyplot as plt

N = 1000
n = np.arange(0,N,1)

s = n[1] - n[0]
f = np.linspace(0,1/s,N)[0:N//2]

'''
1(a)
'''
y_sq = np.piecewise(n,[n>=N/2],[1,0])

plt.figure(figsize=(10,10))
plt.plot(n,y_sq)
plt.title('Square Wave')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.savefig('Ques1/1(a)(i).png')
plt.show()

fft_sq = (np.abs(np.fft.fft(y_sq))**2)[0:N//2]

plt.figure(figsize=(10,10))
plt.plot(f,fft_sq)
plt.ylim(0,500)
plt.title('Square Wave')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.savefig('Ques1/1(a)(ii).png')
plt.show()
'''
1(b)
'''
y_sq = n

plt.figure(figsize=(10,10))
plt.plot(n,y_sq)
plt.ylabel('f(x)')
plt.title('sawtooth wave')
plt.xlabel('x')
plt.grid()
plt.savefig('Ques1/1(b)(i).png')
plt.show()

fft_sq = (np.abs(np.fft.fft(y_sq))**2)[0:N//2]

plt.figure(figsize=(10,10))
plt.plot(f,fft_sq)
plt.title('sawtooth wave')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.savefig('Ques1/1(b)(ii).png')
plt.show()
'''
1(c)
'''
y_sq = np.sin(np.pi*n/N)*np.sin(20*np.pi*n/N)

plt.figure(figsize=(10,10))
plt.plot(n,y_sq)
plt.title('modulated sine wave')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.savefig('Ques1/1(c)(i).png')
plt.show()

fft_sq = (np.abs(np.fft.fft(y_sq))**2)[0:N//2]

plt.figure(figsize=(10,10))
plt.plot(f,fft_sq)
plt.title('modulated sine wave')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.savefig('Ques1/1(c)(ii).png')
plt.show()
