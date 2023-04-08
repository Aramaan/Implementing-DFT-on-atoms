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

def f(t,x,sigma,r,b):
    x, y, z = x
    xdot = sigma*(y-x)
    ydot = r*x - y - x*z
    zdot = x*y - b*z
    return np.array([[xdot,ydot,zdot]]).T

t = np.linspace(0,50,100000)
y0 = [0,1,0]
soln = RK4(lambda t,x: f(t,x,10,28,8/3),y0,t)

plt.figure(figsize=(13,10))
plt.plot(t,soln[1],label=r'$(\sigma,r,b)=(10,28,\frac{8}{3}$)')
plt.grid()
plt.ylabel(r'y')
plt.xlabel(r'time')
plt.title(r'Lorenz Equation')
plt.legend(loc='best',prop={'size': 18})
plt.savefig('Ques3/3(ii).png')
plt.show()

plt.figure(figsize=(13,10))
plt.plot(soln[0],soln[2],label='Initial Condition\n(x,y,z)=(%.1f,%.1f,%.1f)'%(y0[0],y0[1],y0[2]))
plt.grid()
plt.ylabel(r'z')
plt.xlabel(r'x')
plt.title(r'Lorenz Equation Phase Diagram')
plt.legend(loc='best',prop={'size': 18})
plt.savefig('Ques3/3(ii).png')
plt.show()