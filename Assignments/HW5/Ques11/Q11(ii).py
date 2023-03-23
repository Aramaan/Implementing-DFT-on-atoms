

import numpy as np
import matplotlib.pyplot as plt

# Set the size of the grid and the number of particles
grid_size = 101

# Create the grid and initialize the center point
grid = np.zeros((grid_size, grid_size))

# Create a list of particles, starting at the center
c = grid_size // 2
particles = [(c,c)]
grid[c,c] = 1

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
            #print(xdx,ydy)             print(c,c)   
            if grid[xdx, ydy] == 1:
                return True
    return False

# Define a function to check if a particle is touching the edge
def touching_edge(x,y):
    if x == grid_size-1 or x == 0 or y == grid_size-1 or y == 0:
        return True
    return False

def crossing(x,y,r):
    if (x-c)**2 + (y-c)**2 > 4*r**2+1:
        return True
    else:
        return False





r = 0
# Run the simulation
while r <= grid_size/4:
    # Generate a random starting position for the particle
    #x, y = np.random.randint(0, grid_size), np.random.randint(0, grid_size) 
    angle = np.random.uniform(0,2*np.pi)
    x, y = int(np.ceil(c+(r+1)*np.cos(angle))), int(np.ceil(c+(r+1)*np.sin(angle))) 
    #print(r)
    while not (touching_cluster(x, y) or crossing(x,y,r)):
        # Move the particle randomly until it touches the cluster
        dir = np.random.randint(1,5)

        if dir == 1: x+= 1
        if dir == 2: x-= 1
        if dir == 3: y+= 1
        if dir == 4: y-= 1
        #print(x-c,y-c)
    # Add the particle to the cluster

    if crossing(x,y,r):
        continue
    grid[x, y] = 1
    rt = np.sqrt((x-c)**2+(y-c)**2)
    if (rt > r):
        r = rt
    particles.append((x, y))

# Plot the cluster
fig, ax = plt.subplots()
ax.imshow(grid)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(r'Ques11/Q11(ii).png')
plt.show()
