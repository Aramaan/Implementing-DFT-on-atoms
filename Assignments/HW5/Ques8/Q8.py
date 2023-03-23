import numpy as np
from matplotlib import pyplot as plt

def simulated_annealing_mcmc(func, x0, sigma, T0, Tf, cooling_schedule):
    """
    Simulated Annealing using Metropolis-Hastings algorithm.
    func: the objective function to be optimized
    x0: initial value
    sigma: standard deviation for the Gaussian proposal distribution
    T0: initial temperature
    Tf: final temperature
    cooling_schedule: function that takes in temperature and returns new temperature
    """
    xarr = []
    farr = []
    x = x0
    T = T0
    t = 0
    best_x = x
    best_f = func(x)

    while T > Tf:
        # sample a new point from proposal distribution
        x_new = x + np.random.normal(0, sigma)
        f_new = func(x_new)

        # Metropolis-Hastings acceptance probability
        alpha = min(1, np.exp((best_f - f_new) / T))

        # accept or reject the new point
        if np.random.uniform() < alpha:
            x = x_new
            if f_new < best_f:
                best_x = x_new
                best_f = f_new
        t += 1
        xarr.append(x)
        farr.append(f_new)

        # update temperature
        T = cooling_schedule(T)


    return xarr, farr

def f1(x):
    return x**2-np.cos(4*np.pi*x)

c = 1 - 1e-3
def cooling(T0):
    return c*T0
x,f = simulated_annealing_mcmc(f1,2,1,1000,1e-10,cooling)


plt.figure(figsize=(10,10))
plt.plot(x)
plt.title(r'Using Simulated Annealing to find minima of first function')
plt.xlabel(r'step')
plt.ylabel(r'$x$')
plt.grid()
plt.savefig('Ques8/8(i).png')
plt.show()
#print('Minima at: %.1f'%x)
#print('Error estimate: %.2e'%error)

def f2(x):
    return np.cos(x)+np.cos(np.sqrt(2)*x)+np.cos(np.sqrt(3)*x)

c = 1 - 1e-3
def cooling(T0):
    return c*T0
x,f = simulated_annealing_mcmc(f2,25,1,1,1e-10,cooling)

plt.figure(figsize=(10,10))
plt.plot(x)
plt.title(r'Using Simulated Annealing to find minima of second function')
plt.xlabel(r'step')
plt.ylabel(r'$x$')
plt.grid()
plt.savefig('Ques8/8(ii).png')
plt.show()
