import numpy as np
import matplotlib.pyplot as plt

def RK4(ode_func, y0, t):
    """
    Implementation of the Fourth Order Runge-Kutta (RK4) method.
    Args:
        ode_func (function): ODE function f(t, y) to be solved.
        y0 (float): Initial value of the dependent variable.
        t0 (float): Initial value of the independent variable.
        t_end (float): End value of the independent variable.
        h (float): Step size.
    Returns:
        tuple: Tuple containing two arrays: t (array of time steps) and y (array of solutions).
    """
    h = t[1]-t[0]
    n = len(t)
    y = np.zeros([len(y0),n+1])
    y[:,0] = y0

    for i in range(n):
        k1 = h * ode_func(t[i], y[:,i])
        k2 = h * ode_func(t[i] + h/2, y[:,i] + k1[:,0]/2)
        k3 = h * ode_func(t[i] + h/2, y[:,i] + k2[:,0]/2)
        k4 = h * ode_func(t[i] + h, y[:,i] + k3[:,0])

        y[:,i + 1] = y[:,i] + (k1[:,0] + 2*k2[:,0] + 2*k3[:,0] + k4[:,0]) / 6

    return y[:,:n]

import numpy as np
import matplotlib.pyplot as plt

class Projectile:
    def __init__(self, m, R, angle, v0, rho, C, g):
        self.m = m
        self.R = R
        self.angle = angle
        self.v0 = v0
        self.rho = rho
        self.C = C
        self.g = g
        self.x0 = [0, v0 * np.cos(angle), 0, v0 * np.sin(angle)]

    def f_drag(self, t, x):
        x, x1, y, y1 = x
        dxdt = x1
        dydt = y1
        ddxdt = -(np.pi / (2 * self.m)) * self.R ** 2 * self.rho * self.C * dxdt * np.sqrt(dxdt ** 2 + dydt ** 2)
        ddydt = -(np.pi / (2 * self.m)) * self.R ** 2 * self.rho * self.C * dydt * np.sqrt(dxdt ** 2 + dydt ** 2) - self.g
        return np.array([[dxdt, ddxdt, dydt, ddydt]]).T

    def f_no_drag(self, t, x):
        x, x1, y, y1 = x
        dxdt = x1
        dydt = y1
        ddxdt = 0
        ddydt = - self.g
        return np.array([[dxdt, ddxdt, dydt, ddydt]]).T

    def solve_drag(self, t):
        return RK4( self.f_drag, self.x0, t)

    def solve_no_drag(self, t):
        return RK4(self.f_no_drag, self.x0, t)


t = np.linspace(0, 7, 10000)
# Create instance of Projectile class
projectile = Projectile(1, 8 * 1e-2, np.pi/6, 100, 1.22, 0.47, 9.8)

# Solve for trajectory with drag
soln = projectile.solve_drag(t)

# Solve for trajectory without drag
t_no_drag = np.linspace(0, 10.5, 10000)
no_drag = projectile.solve_no_drag(t_no_drag)

plt.figure(figsize=(10,10))
plt.plot(soln[0],soln[2],label=r'With Drag')
plt.plot(no_drag[0],no_drag[2],label=r'Without Drag')
plt.grid()
plt.ylabel(r'y (m)')
plt.xlabel(r'x (m)')
plt.ylim(0,130)
plt.title(r'No Drag Trajectory')
plt.legend(loc='best')
plt.savefig('Ques5/5(i).png')
plt.show()

m = np.linspace(0.1,10,5)
freq = 25
Range = np.zeros(len(m))
t = np.linspace(0,20,1000)

plt.figure(figsize=(10,10))

Range = []
for mass in m:
    projectile = Projectile(mass, 8 * 1e-2, np.pi/6, 100, 1.22, 0.47, 9.8)
    soln = projectile.solve_drag(t)
    range_idx = np.argmax(soln[2, :] < 0) - 1
    range_ = soln[0, range_idx] - soln[0, 0]
    plt.plot(soln[0], soln[2], label=r'mass = %.2f kg, Range = %.2f m' % (mass, range_))
    Range.append(range_)


plt.plot(no_drag[0],no_drag[2],label=r'No Drag Model')
plt.grid()
plt.ylabel(r'y (m)')
plt.xlabel(r'x (m)')
plt.ylim(0,130)
plt.title(r'Quadratic Drag model')
plt.legend(loc='best')
plt.savefig('Ques5/5(ii).png')
plt.show()
    
plt.figure(figsize=(10,10))
plt.plot(m,Range)
plt.grid()
plt.ylabel(r'Range (m)')
plt.xlabel(r'Mass (kg)')
plt.title(r'Mass vs Range')
plt.savefig('Ques5/5(iii).png')
plt.show()