
import numpy as np
import matplotlib.pyplot as plt

L = 101
x = np.arange(0,L,1)
y = np.arange(0,L,1)

x0,y0 = L//2, L//2

steps = int(1e4)
x,y = x0,y0
xarr = np.zeros(steps)
yarr = np.zeros(steps)

for i in range(steps):
    dir = np.random.randint(1,5)
    
    if dir == 1: x+= 1
    if dir == 2: x-= 1
    if dir == 3: y+= 1
    if dir == 4: y-= 1

    if x < 0: x += 1
    if x >= L: x -= 1
    if y < 0: y += 1
    if y >= L: y -= 1

    xarr[i],yarr[i] = x,y

plt.figure(figsize=(10,10))
plt.plot(xarr,yarr)
plt.title(r'Random Walk Simulation')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.grid()
plt.xlim(-1,L)
plt.ylim(-1,L)
plt.savefig('Ques3/Q3.png')
plt.show()