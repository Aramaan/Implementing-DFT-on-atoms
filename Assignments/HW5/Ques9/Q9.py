import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def energy(lattice):
    """
    Computes the energy of a given lattice as minus the number of dimers.
    """
    return -np.sum(lattice)

def metropolis(lattice, T):
    """
    Performs a single Metropolis update on the lattice at temperature T.
    """
    nx, ny = lattice.shape
    i = np.random.randint(nx)
    j = np.random.randint(ny)
    if lattice[i,j] == 1 and lattice[(i+1)%nx,j] == 1:
        # remove dimer
        new_lattice = np.copy(lattice)
        new_lattice[i,j] = 0
        new_lattice[(i+1)%nx,j] = 0
        deltaE = energy(new_lattice) - energy(lattice)
        if deltaE < 0 or np.random.uniform() < np.exp(-deltaE/T):
            lattice[:] = new_lattice[:]
    elif lattice[i,j] == 0 and lattice[(i+1)%nx,j] == 0:
        # add dimer
        new_lattice = np.copy(lattice)
        new_lattice[i,j] = 1
        new_lattice[(i+1)%nx,j] = 1
        deltaE = energy(new_lattice) - energy(lattice)
        if deltaE < 0 or np.random.uniform() < np.exp(-deltaE/T):
            lattice[:] = new_lattice[:]
    # otherwise do nothing
    return np.copy(lattice)

def simulate(lattice, T_init, cooling_schedule, n_steps):
    """
    Simulates the dimer problem using simulated annealing.
    """
    T = T_init
    energies = [energy(lattice)]
    temperatures = [T]
    L = []

    for i in range(n_steps):
        L.append(metropolis(lattice, T))
        T = cooling_schedule(T)
        energies.append(energy(lattice))
        temperatures.append(T)

    return np.array(L), energies, temperatures

# create initial lattice with random dimers
def fill_grid(probability):
    # Initialize empty grid
    grid = np.zeros((50, 50), dtype=np.int8)
    
    # Fill grid randomly with dimers with given probability
    for i in range(49):
        for j in range(49):
            if np.random.rand() < probability:
                grid[i:i+2, j:j+2] = 1
                
    return grid

lattice = fill_grid(0.2)

# define the cooling schedule
T_init = 5.0
n_steps = 10000
cooling_schedule = lambda T: T*0.99

# run the simulation
lattice, energies, temperatures = simulate(lattice, T_init, cooling_schedule, n_steps)

fig = plt.figure(figsize=(10,10))
images = []

for step in np.arange(0,n_steps,10):
    images.append((plt.pcolor(lattice[step,:-1, :-1]),)) 

anime = animation.ArtistAnimation(fig, images, interval=20, repeat_delay=1000, blit=True)
anime.save('Ques9/Dimer.mp4', metadata={})      
plt.show()