import numpy as np
from matplotlib import pyplot as plt


def leapfrog(f, x0, v0, t0, tf, dt):
    """
    Leapfrog method for solving ordinary differential equations (ODEs)
    
    Parameters:
        f: callable
            The function defining the ODE. Should take x and v as inputs and return dv/dt.
        x0: float
            The initial position.
        v0: float
            The initial velocity.
        t0: float
            The initial time.
        tf: float
            The final time.
        dt: float
            The time step.
    
    Returns:
        x: numpy array
            The array of positions at each time step.
        v: numpy array
            The array of velocities at each time step.
        t: numpy array
            The array of times at each time step.
    """
    num_steps = int((tf - t0) / dt)  # Number of time steps
    x = np.zeros(num_steps + 1)  # Array to store positions
    v = np.zeros(num_steps + 1)  # Array to store velocities
    t = np.linspace(t0, tf, num_steps + 1)  # Array to store times

    x[0] = x0
    v[0] = v0

    for i in range(num_steps):
        # Leapfrog update for velocity
        m,n = f(t[i-1],x[i-1],v[i-1])
        xhalf = x[i-1] + (dt/2)*m
        vhalf = v[i-1] + (dt/2)*n
        k,l = f(t[i-1]+dt/2,xhalf,vhalf)
        x[i] = x[i-1] + dt*k
        v[i] = v[i-1] + dt*l

    return x, v, t

def f(t,x,v): 
    xdot = v
    vdot = v**2 - x - 5
    return xdot,vdot

x0,v0 = 1,0
x,v,t = leapfrog(f,x0,v0,0,50,0.001)

plt.figure(figsize=(10,10))
plt.plot(t,x)
plt.title(r'Solution (Leapfrog)')
plt.grid()
plt.ylabel(r'x')
plt.xlabel(r't')
plt.savefig('Ques9/9(ii).png')
plt.show()
