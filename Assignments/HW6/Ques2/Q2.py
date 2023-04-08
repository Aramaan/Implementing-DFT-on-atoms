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

def f(t,x,alpha,beta,gamma,delta):
    x, y = x
    xdot = alpha*x-beta*x*y
    ydot = gamma*x*y - delta*y
    return np.array([[xdot,ydot]]).T

t = np.linspace(0,30,100000)
y0 = [2,2]
pop = RK4(lambda t,x: f(t,x,1,0.5,0.5,2.),y0,t)

plt.figure(figsize=(10,10))
plt.plot(t,pop[0],label = 'Prey Population')
plt.grid()
plt.plot(t,pop[1],label='Predator Population')
plt.legend(loc='best',prop={'size': 18})
plt.ylabel(r'Population')
plt.xlabel(r'time')
plt.title(r'Lotka Volterra Model')
plt.savefig('Ques2/2(i).png')
plt.show()

plt.figure(figsize=(10,10))
plt.plot(pop[0],pop[1],label='Initial Population\nPredator: %d Prey: %d'%(int(y0[1]),int(y0[0])))
plt.grid()
plt.ylabel(r'Prey Population')
plt.xlabel(r'Predator Population')
plt.title(r'Lotka Volterra Model Phase Diagram')
plt.legend(loc='best',prop={'size': 18})
plt.savefig('Ques2/2(ii).png')
plt.show()