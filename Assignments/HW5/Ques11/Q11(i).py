

import numpy as np
import matplotlib.pyplot as plt

# Set the size of the grid and the number of particles
grid_size = 101

# Create the grid and initialize the center point
grid = np.zeros((grid_size, grid_size))

# Create a list of particles, starting at the center
particles = []

# Define a function to check if a particle is touching the cluster
def touching_cluster(x, y):
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            xdx = x + dx
            ydy = y + dy
            if xdx == grid_size or ydy == grid_size or xdx == -1 or ydy == -1:
                continue    
            if grid[xdx, ydy] == 1:
                return True
    return False

# Define a function to check if a particle is touching the edge
def touching_edge(x,y):
    if x == grid_size-1 or x == 0 or y == grid_size-1 or y == 0:
        return True
    return False
c = grid_size // 2
# Run the simulation
while grid[c, c] != 1:
    # Generate a random starting position for the particle
    #x, y = np.random.randint(0, grid_size), np.random.randint(0, grid_size)   
    x,y = c, c
    while not (touching_cluster(x, y) or touching_edge(x,y)):
        # Move the particle randomly until it touches the cluster
        dir = np.random.randint(1,5)
        if dir == 1: x+= 1
        if dir == 2: x-= 1
        if dir == 3: y+= 1
        if dir == 4: y-= 1
    # Add the particle to the cluster
    grid[x, y] = 1
    particles.append((x, y))

# Plot the cluster
fig, ax = plt.subplots()
ax.imshow(grid)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(r'Ques11/Q11(i).png')
plt.show()
