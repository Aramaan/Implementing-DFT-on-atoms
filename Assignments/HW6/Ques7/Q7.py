import numpy as np
import matplotlib.pyplot as plt

def RK4(ode_func, y0, t):
    """
    Implementation of the Fourth Order Runge-Kutta (RK4) method.
    Args:
        ode_func (function): ODE function f(t, y) to be solved.
        y0 (float): Initial value of the dependent variable.
        t0 (float): Initial value of the independent variable.
        t_end (float): End value of the independent variable.
        h (float): Step size.
    Returns:
        tuple: Tuple containing two arrays: t (array of time steps) and y (array of solutions).
    """
    h = t[1]-t[0]
    n = len(t)
    y = np.zeros([len(y0),n+1])
    y[:,0] = y0

    for i in range(n):
        k1 = h * ode_func(t[i], y[:,i])
        k2 = h * ode_func(t[i] + h/2, y[:,i] + k1[:,0]/2)
        k3 = h * ode_func(t[i] + h/2, y[:,i] + k2[:,0]/2)
        k4 = h * ode_func(t[i] + h, y[:,i] + k3[:,0])

        y[:,i + 1] = y[:,i] + (k1[:,0] + 2*k2[:,0] + 2*k3[:,0] + k4[:,0]) / 6

    return y[:,:n]

def f(t,x,m,k,F,N): #F is an array of functions of time
    pos = x[0::2]
    vel = x[1::2]
    acc = np.zeros(N)
    for i in range(N-1):
        acc[i] = (k/m)*(pos[i+1]-pos[i])+F[i](t)
    acc[-1] = (k/m)*(pos[N-2]-pos[len(pos)-1])+F[N-1](t)
    vector = np.zeros((2*N,1))
    vector[0::2,0] = vel
    vector[1::2,0] = acc
    return vector

F = [lambda t:0 for i in range(0,5)]
F[0] = lambda t: np.cos(2*t)

t = np.linspace(0,20,10000)
y0 =  np.random.randint(0,10,size=2*5)
soln = RK4(lambda t,x: f(t,x,1,6,F,5),y0,t)

plt.figure(figsize=(10,10))
for i in range(0,5):
    plt.plot(t,soln[2*i],label=r'Mass No. %d'%(i+1))
plt.grid()
plt.ylabel(r'$x$')
plt.xlabel(r'$t$')
plt.title(r'Coupled oscillator')
plt.legend(loc='best')
plt.savefig('Ques7/7.png')
plt.show()