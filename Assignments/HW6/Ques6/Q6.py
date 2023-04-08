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


def f(t,z,M,L): #Quadratic drag model
    x, x1, y, y1 = z
    r = np.sqrt(x**2+y**2)
    denom = r**2*np.sqrt(r**2+(L/2)**2)
    dxdt = x1
    dydt = y1
    dx1dt = -M * x/denom
    dy1dt = -M * y/denom
    return np.array([[dxdt,dx1dt,dydt,dy1dt]]).T

t = np.linspace(0,10,100000)
y0 = [1,0,0,1] 
M = 10
L = 2
x,vx,y,vy = RK4(lambda t,y: f(t,y,M,L),y0,t)

plt.figure(figsize=(10,10))
plt.plot(x,y)
plt.title(r'Trajectory of the space garbage ')
plt.grid()
plt.ylabel(r'y')
plt.xlabel(r'x')
plt.savefig('Ques6/6.png')
plt.show()