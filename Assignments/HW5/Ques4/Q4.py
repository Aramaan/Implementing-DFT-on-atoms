
import numpy as np
import matplotlib.pyplot as plt

N = int(1e4)
x = np.linspace(0,2,N)
def func(x):
    return (np.sin(1/(x*(2-x))))**2
f = func(x)

plt.figure(figsize=(10,10))
plt.plot(x,f)
plt.title(r'Integrand')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.grid()
plt.savefig('Ques4/Q4.png')
plt.show()

x = np.random.uniform(0,2,size = N)
y = np.random.uniform(0,1,size = N)
Area = 2
I = np.count_nonzero(y<f)*Area/N
e = np.sqrt(I*(Area-I)/N)
print('The value of the integral through Monte Carlo Hit-Miss Algorithm is {} and the error is {}'.format(I,e))

f = func(x)
I = (2/N)*np.sum(f)
variance = (1/N)*np.sum(func(x)**2)-(I/2)**2
error = (2.-0.)*np.sqrt(variance/N)
print('The value of the integral through Monte Carlo Mean Value Algorithm is {} and the error is {}'.format(I,e))
