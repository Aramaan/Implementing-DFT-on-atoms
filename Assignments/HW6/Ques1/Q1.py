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

Vout0 = 0
int()
def Input(t):
    vector = np.vectorize(np.int_)
    x = vector(np.floor(2*t))
    I = np.piecewise(t,x%2==0,[1,-1])
    return I


def dVdt(t,Vout,RC):
    V = Input(t)
    return (V - Vout)/RC

t = np.linspace(0,10,100000)

plt.figure(figsize=(10,10))
plt.plot(t,np.array([Input(time) for time in t]))
plt.grid()
plt.ylabel(r'Input Voltage')
plt.xlabel(r'time')
plt.title(r'Low Pass Filter (Input Voltage)')
plt.savefig('Ques1/input.png')
plt.show()

Vout = RK4(lambda t,x: dVdt(t,x,0.01),Vout0,t)

plt.figure(figsize=(10,10))
plt.plot(t,Vout)
plt.grid()
plt.ylabel(r'Output Voltage')
plt.xlabel(r'time')
plt.title(r'Low Pass Filter ($RC=%0.01$)')
plt.savefig('Ques1/1(i).png')
plt.show()

Vout = RK4(lambda t,x: dVdt(t,x,0.1),Vout0,t)

plt.figure(figsize=(10,10))
plt.plot(t,Vout)
plt.grid()
plt.ylabel(r'Output Voltage')
plt.xlabel(r'time')
plt.title(r'Low Pass Filter ($RC=%0.1$)')
plt.savefig('Ques1/1(ii).png')
plt.show()

Vout = RK4(lambda t,x: dVdt(t,x,1),Vout0,t)

plt.figure(figsize=(10,10))
plt.plot(t,Vout)
plt.grid()
plt.ylabel(r'Output Voltage')
plt.xlabel(r'time')
plt.title(r'Low Pass Filter ($RC=%1$)')
plt.savefig('Ques1/1(iii).png')
plt.show()