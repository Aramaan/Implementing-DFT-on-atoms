import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G

# Constants
M_s = 1.989e30     # Mass of the Sun (kg)
M_e = 5.972e24     # Mass of the Earth (kg)
AU = 1.496e11      # Astronomical Unit (m)
year = 3.154e7     # Year in seconds

# Initial conditions
x0 = AU          
y0 = 0            
vx0 = 0          
vy0 = 3e4         

# Time step and number of steps
dt = year / 365    # Time step (s)
num_steps = 365*5  # Number of steps (simulate 5 years)

# Initialize arrays to store positions, velocities, accelerations, potential energy, and kinetic energy
t = np.zeros(num_steps)
x = np.zeros(num_steps)  # Array to store x-coordinates
y = np.zeros(num_steps)  # Array to store y-coordinates
vx = np.zeros(num_steps) # Array to store x-velocities
vy = np.zeros(num_steps) # Array to store y-velocities
ax = np.zeros(num_steps) # Array to store x-accelerations
ay = np.zeros(num_steps) # Array to store y-accelerations
PE = np.zeros(num_steps) # Array to store potential energy
KE = np.zeros(num_steps) # Array to store kinetic energy

# Set initial conditions
t[0] = 0
x[0] = x0
y[0] = y0
vx[0] = vx0
vy[0] = vy0

# Verlet integration loop
for i in range(1, num_steps):
    #Update time
    t[i] = t[i-1] + 1


    # Update x-coordinate
    x[i] = x[i-1] + vx[i-1]*dt + 0.5*ax[i-1]*dt**2 
    
    # Update y-coordinate
    y[i] = y[i-1] + vy[i-1]*dt + 0.5*ay[i-1]*dt**2
    
    # Calculate distance between the Earth and the Sun
    r = np.sqrt(x[i]**2 + y[i]**2)
    
    # Calculate acceleration of the Earth due to gravitational force
    ax[i] = -G * M_s * x[i] / r**3
    ay[i] = -G * M_s * y[i] / r**3
    
    # Update x-velocity
    vx[i] = vx[i-1] + 0.5 * (ax[i] + ax[i-1]) * dt
    
    # Update y-velocity
    vy[i] = vy[i-1] + 0.5 * (ay[i] + ay[i-1]) * dt
    
    # Calculate potential energy
    PE[i] = -G * M_s * M_e / r
    
    # Calculate kinetic energy
    KE[i] = 0.5 * M_e * (vx[i]**2 + vy[i]**2)

# Plot the orbit of the Earth around the Sun

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('x-coordinate (m)')
plt.ylabel('y-coordinate (m)')
plt.title('Orbit of Earth around Sun')
plt.axis('equal')
plt.grid()
plt.grid()
plt.savefig('Ques10/10(i).png')
plt.show()


t = t[1:]
plt.plot(t,KE[1:])
plt.plot(t,PE[1:])
plt.plot(t,KE[1:]+PE[1:])
plt.xlabel('time in days')
plt.ylabel('Energy')
plt.title('Variations of Energies with time')
plt.legend(['Kinetic Energy','Potential Energy','Total Energy'])
plt.grid()
plt.grid()
plt.savefig('Ques10/10(ii).png')
plt.show()


plt.plot(t,KE[1:]+PE[1:])
plt.xlabel('time in days')
plt.ylabel('Total Energy')
plt.title('Total Energy')
plt.grid()
plt.savefig('Ques10/10(iii).png')
plt.show()
