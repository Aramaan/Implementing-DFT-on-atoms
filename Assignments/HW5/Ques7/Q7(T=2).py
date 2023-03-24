
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

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

    x_old = np.copy(x)
    x_new = np.copy(x)
    f_old = E(x)

    for i in range(steps):
        # sample a new point from proposal distribution
        rx = np.random.randint(0,nx)
        ry = np.random.randint(0,ny)
        x_new[rx,ry]  *= -1
        
        s = np.hstack((np.take(x_new[rx],[ry-1,ry+1],mode='wrap'),np.take(x_new[:,ry],[rx-1,rx+1],mode='wrap')))
        f_new = f_old-2*(-J*x_old[rx,ry]*np.sum(s))

        # Metropolis-Hastings acceptance probability
        alpha = min(1, np.exp((f_old - f_new) / T))

        # accept or reject the new point
        if np.random.uniform() < alpha:
            x_old = np.copy(x_new)
            f_old = f_new
        else:
            x_new = np.copy(x_old)
            f_new = f_old



        xarr.append(x_old)
        marr.append(M(x_old))
        farr.append(f_old)

    return np.array(xarr), farr, marr

        # update temperature

J = 1
def Energy(L):
    sum = 0
    n1,n2 = L.shape
    for i in range(n1):
        for j in range(n2):
            s = np.hstack((np.take(L[i],[j-1,j+1],mode='wrap'),np.take(L[:,j],[i-1,i+1],mode='wrap')))
            sum += L[i,j]*np.sum(s)
    E = -J*(sum/2)
    return E

def Mag(L):
    sum = np.sum(L)
    return sum


T = 2
nx,ny = 20,20
L = np.random.choice([1,-1],size = [nx,ny])
steps = int(1e6)
L,E,M = metropolis_mcmc(L, Energy, Mag, steps, T)


plt.figure(figsize=(10,10))
plt.pcolor(L[-1])
plt.title(r'Final State of Ising model')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.savefig('Ques7/7(T=2)(i)T1.png')
plt.show()


plt.figure(figsize=(10,10))
plt.plot(np.arange(0,steps,1),E)
plt.title(r'Energy of the system')
plt.xlabel(r'step')
plt.ylabel(r'Energy')
plt.grid()
plt.savefig('Ques7/7(T=2)(ii)T1.png')
plt.show()

plt.figure(figsize=(10,10))
plt.plot(np.arange(0,steps,1),M)
plt.title(r'Magnetization of the system')
plt.xlabel(r'step')
plt.ylabel(r'Magnetization')
plt.grid()
plt.savefig('Ques7/7(T=2)(iii)T1.png')
plt.show()

fig = plt.figure(figsize=(10,10))
images = []

# for step in np.arange(0,steps,1):
#     ims.append(plt.pcolor(L[step])) 


for step in np.arange(0,steps//10,100):
    images.append((plt.pcolor(L[step,:-1, :-1]),)) 

anime = animation.ArtistAnimation(fig, images, interval=20, repeat_delay=1000, blit=True)
anime.save('Ques7/Simulation(T=2).mp4', metadata={})    
plt.show()

'''
When a magnetic material is cooled below its critical temperature (Tc), all the spins in the system have
 a tendency to align in the same direction. Depending on the initial configuration of the Monte Carlo random walkers, 
 the system can achieve either a positive net magnetization (all spins up) or a negative net magnetization (all spins down). 
 However, since the choice of coordinates used to denote spin up or down is arbitrary, these two configurations are
   indistinguishable from a physical perspective. The relevant factor is the relative orientation of the individual spins of the atoms.

As the temperature rises above the critical temperature, a phase transition occurs, and the system loses
 its spontaneous magnetization. At this point, the spins of the atoms begin adopting random relative orientations, 
 causing the material to transform from a ferromagnetic state to a paramagnetic state. In the paramagnetic state,
   the material only exhibits magnetic properties in the presence of an external magnetic field, and its magnetic 
   moments become oriented randomly in the absence of a field.
'''