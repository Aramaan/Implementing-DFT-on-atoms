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
    y = np.zeros(n+1)
    y[0] = y0

    for i in range(n):
        k1 = h * ode_func(t[i], y[i])
        k2 = h * ode_func(t[i] + h/2, y[i] + k1/2)
        k3 = h * ode_func(t[i] + h/2, y[i] + k2/2)
        k4 = h * ode_func(t[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return y[:n]

'''
Harmonic Oscillator
'''

def f(t,x,omega):
    x, y = x
    xdot = y
    ydot = -omega**2*x 
    return np.array([[xdot,ydot]]).T

x0 = [1,0] #x(t=0), dx/dt(t=0)
t = np.linspace(0,50,1000)
omega = 1
soln = RK4(f,x0,t,omega)

plt.figure(figsize=(13,10))
plt.plot(t,soln[0])#,label=r'$(\sigma,r,b)=(10,28,\frac{8}{3}$)')
plt.grid()
plt.ylabel(r'x',size=18)
plt.xlabel(r'time',size=20)
plt.title(r'Simple Harmonic Oscillator $\ddot{x}=-\omega ^2 x$', size=21)
plt.savefig('4_1.png')
plt.show()

plt.figure(figsize=(13,10))
plt.plot(soln[0],soln[1])#,label=r'$(\sigma,r,b)=(10,28,\frac{8}{3}$)')
plt.grid()
plt.ylabel(r'Velocity',size=18)
plt.xlabel(r'Displacement',size=20)
plt.title(r'Simple Harmonic Oscillator Phase Diagram$', size=21)
plt.savefig('4_2.png')
plt.show()

sampling = 1/(t[1]-t[0])
period = 1/(np.argmax(np.abs(np.fft.fft(soln[0]))[:len(t)//2])*sampling/len(t))

Amplitude = np.linspace(1,10,100)
period = [RK4.RK4(f,[A,0],t,omega)[0] for A in Amplitude]
period = [1/(np.argmax(np.abs(np.fft.fft(x))[:len(t)//2])*sampling/len(t)) for x in period]


plt.figure(figsize=(13,10))
plt.plot(Amplitude,period)#,label=r'$(\sigma,r,b)=(10,28,\frac{8}{3}$)')
plt.grid()
plt.ylabel(r'Time Period',size=18)
plt.xlabel(r'Amplitude',size=20)
plt.title(r'Simple Harmonic Oscillator $\ddot{x}=-\omega ^2 x$', size=21)
plt.savefig('4_3.png')
plt.show()

'''
Anharmonic Oscillator
'''

def f(t,x,omega):
    x, y = x
    xdot = y
    ydot = -omega**2*x**3
    return np.array([[xdot,ydot]]).T

x0 = [1,0] #x(t=0), dx/dt(t=0)
t = np.linspace(0,50,1000)
omega = 1
soln = RK4(f,x0,t,omega)

plt.figure(figsize=(13,10))
plt.plot(soln[0],soln[1])#,label=r'$(\sigma,r,b)=(10,28,\frac{8}{3}$)')
plt.grid()
plt.ylabel(r'Velocity',size=18)
plt.xlabel(r'Displacement',size=20)
plt.title(r'Anharmonic Oscillator Phase Diagram$', size=21)
plt.savefig('4_4.png')
plt.show()

Amplitude = np.linspace(1,10,100)
period = [RK4.RK4(f,[A,0],t,omega)[0] for A in Amplitude]
period = [1/(np.argmax(np.abs(np.fft.fft(x))[:len(t)//2])*sampling/len(t)) for x in period]


plt.figure(figsize=(13,10))
plt.plot(Amplitude,period)#,label=r'$(\sigma,r,b)=(10,28,\frac{8}{3}$)')
plt.grid()
plt.ylabel(r'Time Period',size=18)
plt.xlabel(r'Amplitude',size=20)
plt.title(r'Anharmonic Oscillator $\ddot{x}=-\omega ^2 x^3$', size=21)
plt.savefig('4_5.png')
plt.show()

'''
van der Pol oscillator
'''

def f(t,x,omega,mu):
    x, y = x
    xdot = y
    ydot = mu*(1-x**2)*y-omega**2*x
    return np.array([[xdot,ydot]]).T

x0 = [1,0] #x(t=0), dx/dt(t=0)
t = np.linspace(0,20,100000)
omega, mu = 1, [1,2,4,5]

plt.figure(figsize=(13,10))
for i in range(0,len(mu)):
    soln = RK4.RK4(f,x0,t,omega,mu[i])
    plt.plot(soln[0],soln[1],label=r'$(\omega,\mu)$=(%d,%d)'%(omega,mu[i]))
    
plt.grid()
plt.ylabel(r'Velocity',size=18)
plt.xlabel(r'Displacement',size=20)
plt.title(r'Van der Pol Oscillator Phase Diagram', size=21)
plt.legend(loc='best',prop={'size': 18})
plt.savefig('4_6.png')
plt.show()
    