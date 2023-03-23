import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import datetime

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

fig = plt.figure(figsize=(10,10))
ims = []

for add in np.arange(0,pos.shape[0]//4,1):
    ims.append((plt.pcolor(pos[add,:-1, :-1],cmap='nipy_spectral'),)) 

im_ani = animation.ArtistAnimation(fig, ims, interval=35, repeat_delay=1000, blit=True)
im_ani.save('Dimer Filling.mp4', metadata={'Artist':'Alankar','Album':'PH 354','Comment':'steps = %d'%steps,'Title':'Dimer Covering Problem','Year':datetime.datetime.now().year})
plt.show()