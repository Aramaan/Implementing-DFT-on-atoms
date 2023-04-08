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

def f(t,x,m,k,F): #F is an array of functions of time
    pos = [x[i] for i in range(0,len(x),2)]
    vel = [x[i] for i in range(1,len(x),2)]
    acc = [(k/m)*(pos[i+1]-pos[i])+F[i](t) for i in range (0,len(pos)-1)]
    acc.append((k/m)*(pos[len(pos)-2]-pos[len(pos)-1])+F[len(pos)-1](t))
    vector = np.zeros((2*len(pos),1))
    counter = 0
    for i in range(0,2*len(pos),2):
        vector[i] = vel[counter]
        vector[i+1] = acc[counter]
        counter += 1
    return vector

N = 5
m = 1
k = 6
omega = 2

F = [lambda t:0 for i in range(0,N)]
F[0] = lambda t: np.cos(omega*t)

t = np.linspace(0,20,10000)
y0 =  np.random.randint(0,10,size=2*N)
soln = RK4(lambda t,x: f(t,x,m,k,F),y0,t)

plt.figure(figsize=(10,10))
for i in range(0,N):
    plt.plot(t,soln[2*i],label=r'Mass No. %d'%(i+1))
plt.grid()
plt.ylabel(r'$x$')
plt.xlabel(r'$t$')
plt.title(r'Coupled oscillator')
plt.legend(loc='best')
plt.savefig('Ques7/7.png')
plt.show()