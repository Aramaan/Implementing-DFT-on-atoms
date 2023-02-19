import numpy as np
import matplotlib.pyplot as plt

def func(x,c):
    return 1-np.exp(-c*x)

def FPI(f,x0,error):
    xi = x0
    xf = x0 +1
    c = 0
    while(np.abs(xf-xi)>error):
        c += 1
        xi = xf
        xf = f(xf)
    return c, xf

def AFPI(func,x0,error,*args):
    xi = x0
    xf = x0 + 10
    c = 0
    while(np.abs(xf-xi)>error):
        c += 1
        xi = xf
        g = (func(func(xi,*args),*args)-func(xi,*args))/(func(xi,*args)-xi)-1
        xf = xi-(func(xi,*args)-xi)/g
    return c, xf

a,b = FPI(lambda x: func(x,2),1,1e-6)
print("for c = 2 the solution of the equation that converged in {} steps is {} with accuracy of {}".format(a,b,1e-6))

'''
for c = 2 the solution of the equation that converged in 13 steps is 0.7968126311118457 with accuracy of 1e-06
'''
L = []
C = np.arange(0,3,0.01)
for c in C:
    a,b = FPI(lambda x: func(x,c),1,1e-6)
    L.append(b)

plt.plot(C,L)
plt.show()

a,b = AFPI(lambda x: func(x,2),1,1e-6)
print("for c = 2 the solution of the equation that converged in {} steps is {} with accuracy of {}".format(a,b,1e-6))

'''
for c = 2 the solution of the equation that converged in 13 steps is 0.7968126311118457 with accuracy of 1e-06
'''
