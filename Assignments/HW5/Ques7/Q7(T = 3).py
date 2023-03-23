
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime

def metropolis_mcmc(Lattice, E, M, steps, T0):
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
    marr = []
    x = Lattice
    T = T0
    t = 0

    x_new = np.copy(x)
    best_f = E(x)

    for i in range(steps):
        # sample a new point from proposal distribution
        rx = np.random.randint(0,nx)
        ry = np.random.randint(0,ny)
        x_new[rx,ry]  *= -1
        f_new = E(x_new)

        # Metropolis-Hastings acceptance probability
        alpha = min(1, np.exp((best_f - f_new) / T))

        # accept or reject the new point
        if np.random.uniform() < alpha:
            x = np.copy(x_new)
            if f_new < best_f:
                best_f = f_new
        t += 1
        xarr.append(x)
        marr.append(M(x))
        farr.append(f_new)

    return np.array(xarr), farr, marr

        # update temperature

J = 1
def Energy(L):
    sum = 0
    n1,n2 = L.shape
    for i in range(n1):
        for j in range(n2):
            s = np.hstack((np.take(L[i],[j-1,j+1],mode='wrap'),np.take(L[:,j],[i-1,i+1],mode='wrap')))
            sum += np.sum(L[i,j]*s)
    E = -J*(sum/4)
    return E

def Mag(L):
    sum = 0
    n1,n2 = L.shape
    for i in range(n1):
        for j in range(n2):
            sum += L[i,j]
    return sum


T = 3
nx,ny = 20,20
L = np.random.choice([1,-1],size = [nx,ny])
steps = int(1e3)
L,E,M = metropolis_mcmc(L, Energy, Mag, steps, T)


plt.figure(figsize=(10,10))
plt.pcolor(L[-1])
plt.title(r'Final State of Ising model')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.savefig('Ques7/7(T=3)(i)T1.png')
plt.show()


plt.figure(figsize=(10,10))
plt.plot(np.arange(0,steps,1),E)
plt.title(r'Energy of the system')
plt.xlabel(r'step')
plt.ylabel(r'Energy')
plt.grid()
plt.savefig('Ques7/7(T=3)(ii)T1.png')
plt.show()

plt.figure(figsize=(10,10))
plt.plot(np.arange(0,steps,1),M)
plt.title(r'Magnetization of the system')
plt.xlabel(r'step')
plt.ylabel(r'Magnetization')
plt.grid()
plt.savefig('Ques7/7(T=3)(iii)T1.png')
plt.show()

fig = plt.figure(figsize=(10,10))
images = []

# for step in np.arange(0,steps,1):
#     ims.append(plt.pcolor(L[step])) 


for step in np.arange(0,steps,1):
    images.append((plt.pcolor(L[step,:-1, :-1]),)) 

anime = animation.ArtistAnimation(fig, images, interval=30, repeat_delay=1000, blit=True)
anime.save('Ques7/Simulation(T=3).mp4', metadata={})    
plt.show()