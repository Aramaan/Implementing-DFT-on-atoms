import numpy as np

N = int(1e6)
f = lambda x: (1/np.sqrt(x))/(np.exp(x)+1)
w = lambda x:1/(np.sqrt(x)*2)

x = np.random.uniform(0,1,size=N)**2
#transformation rule is x = u**2
k = f(x)/w(x)
I = (1/N)*np.sum(k)

variance = (1/N)*np.sum((k)**2)-(I)**2
e = np.sqrt(variance/N)

print('The Value of the Integral through importance sampling monte carlo \n is {} with an error estimate of {}'.format(I,e))




