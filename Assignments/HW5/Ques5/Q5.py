import numpy as np

N = int(1e6)
d = 10
def Rsquare(N,d):
    x = np.zeros([d,N])
    for i in range(10):
        x[i,:] = np.array(np.random.uniform(-1,1,size = N))
    return sum(map(lambda x: x**2,x))
f = Rsquare(N,d) 

Volume = 2**d
I = np.count_nonzero(f<1)*Volume/N
e = np.sqrt(I*(Volume-I)/N)
print('The volume of unit sphere through Monte Carlo Hit-Miss Algorithm is {} and the error is {}'.format(I,e)) 

f = func(x)
I = (2/N)*np.sum(f)
variance = (1/N)*np.sum(func(x)**2)-(I/2)**2
error = (2.-0.)*np.sqrt(variance/N)
print('The value of the integral through Monte Carlo Mean Value Algorithm is {} and the error is {}'.format(I,e))