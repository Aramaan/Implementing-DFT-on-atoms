import numpy as np
import matplotlib.pyplot as plt

# Time step in seconds
dt = 1 

# Half-life values in minutes
tau1, tau2, tau3 = 46, 2.2, 3.3 

def decay_type1(N1, N2, N3, N4):
    """Function to simulate change of first isotope"""
    change = np.random.uniform(0,1, size=N1)
    prob = 1 - 2**(-dt / (tau1 * 60))
    change = int(np.sum(np.array(change <= prob )))
    N1 -= change
    N2 += change
    return (N1, N2, N3, N4)

def decay_type2(N1, N2, N3, N4):
    """Function to simulate change of second isotope"""
    change = np.random.uniform(0,1, size=N3)
    prob = 1 - 2**(-dt / (tau2 * 60))
    change = int(np.sum(np.array(change <= prob )))
    N3 -= change
    N1 += change
    return (N1, N2, N3, N4)

def decay_type3(N1, N2, N3, N4):
    """Function to simulate change of third isotope"""
    route = np.random.uniform(0,1, size=N3)
    N1_to_N2 = int(np.sum(route < 0.9791))
    N1_to_N4 = N3 - N1_to_N2 
    # change to N2
    change = np.random.uniform(0,1, size=N1_to_N2)
    prob = 1 - 2**(-dt / (tau3 * 60))
    change = int(np.sum(np.array(change <= prob )))
    N3 -= change
    N2 += change
    # change to N4
    change = np.random.uniform(0,1, size=N1_to_N4)
    prob = 1 - 2**(-dt / (tau3 * 60))
    change = int(np.sum(np.array(change <= prob )))
    N3 -= change
    N4 += change
    return (N1, N2, N3, N4)

# Number of time steps and initialization of arrays for each isotope
t = 20000
steps = t // dt    
N1, N2, N3, N4 = np.zeros(steps+1,dtype=np.int16),np.zeros(steps+1,dtype=np.int16),np.zeros(steps+1,dtype=np.int16),np.zeros(steps+1,dtype=np.int16) 
N1[0], N2[0], N3[0], N4[0] = 0, 0, 10000, 0

# Simulation loop
for i in range(1, steps+1):
    N1[i], N2[i], N3[i], N4[i] = decay_type1(N1[i-1], N2[i-1], N3[i-1], N4[i-1])
    N1[i], N2[i], N3[i], N4[i] = decay_type2(N1[i], N2[i], N3[i], N4[i])
    N1[i], N2[i], N3[i], N4[i] = decay_type3(N1[i], N2[i], N3[i], N4[i])

# Plot results
plt.plot(N1)
plt.plot(N2)
plt.plot(N3)
plt.plot(N4)
plt.legend(['Pb', 'Bi209', 'Bi213', 'Tl'])
plt.xlabel('Time')
plt.ylabel('Count')
plt.savefig('Ques2/Q2.png')
plt.show()

