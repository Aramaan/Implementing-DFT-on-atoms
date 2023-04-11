import numpy as np
import matplotlib.pyplot as plt

def RK4(f, x0, t, *args):
    dt = t[1] - t[0]
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(len(t) - 1):
        k1 = dt * f(t[i], x[i], *args)
        k2 = dt * f(t[i] + dt/2, x[i] + k1/2, *args)
        k3 = dt * f(t[i] + dt/2, x[i] + k2/2, *args)
        k4 = dt * f(t[i] + dt, x[i] + k3, *args)
        x[i+1] = x[i] + (k1 + 2*k2 + 2*k3 + k4)/6
    return x.T

# Define the harmonic oscillator function
def harmonic_oscillator(t, x, omega):
    x, y = x
    dxdt = y
    dydt = -omega**2 * x
    return np.array([dxdt, dydt])

# Define the anharmonic oscillator function
def anharmonic_oscillator(t, x, omega):
    x, y = x
    dxdt = y
    dydt = -omega**2 * x**3
    return np.array([dxdt, dydt])

# Define the van der Pol oscillator function
def van_der_pol_oscillator(t, x, omega, mu):
    x, y = x
    dxdt = y
    dydt = mu * (1 - x**2) * y - omega**2 * x
    return np.array([dxdt, dydt])

'''
Harmonic Oscillator
'''

x0 = [1,0] #x(t=0), dx/dt(t=0)
t = np.linspace(0,50,1000)
omega = 1
soln = RK4(harmonic_oscillator,x0,t,omega)

plt.figure(figsize=(10,10))
plt.plot(t,soln[0])
plt.grid()
plt.ylabel('Displacement (x)')
plt.xlabel('Time')
plt.title('Simple Harmonic Oscillator: Displacement vs Time')
plt.savefig('Ques4/4(i).png')
plt.show()

plt.figure(figsize=(10,10))
plt.plot(soln[0],soln[1])
plt.grid()
plt.ylabel('Velocity')
plt.xlabel('Displacement (x)')
plt.title('Simple Harmonic Oscillator: Phase Diagram')
plt.savefig('Ques4/4(ii).png')
plt.show()

sampling = 1/(t[1]-t[0])
freqs = np.fft.fftfreq(len(t), d=1/sampling)
ft_x = np.fft.fft(soln[0])
ft_x_abs = np.abs(ft_x[:len(t)//2])

period = 1 / freqs[np.argmax(ft_x_abs)]
amplitude = np.linspace(1,10,100)

plt.figure(figsize=(10,10))
plt.plot(amplitude, period * np.ones(len(amplitude)))
plt.grid()
plt.ylabel('Time Period')
plt.xlabel('Amplitude')
plt.title('Simple Harmonic Oscillator: Time Period vs Amplitude')
plt.savefig('Ques4/4(iii).png')
plt.show()

'''
Anharmonic Oscillator
'''

x0 = [1,0] 
t = np.linspace(0,50,1000)
omega = 1
sol = RK4(anharmonic_oscillator,x0,t,omega)

plt.figure(figsize=(10,10))
plt.plot(sol[0],sol[1])
plt.grid()
plt.ylabel(r'Velocity')
plt.xlabel(r'Displacement')
plt.title(r'Anharmonic Oscillator: Phase Diagram$')
plt.savefig('Ques4/4(iv).png')
plt.show()

sampling = 1/(t[1]-t[0])
freqs = np.fft.fftfreq(len(t), d=1/sampling)
ft_x = np.fft.fft(sol[0])
ft_x_abs = np.abs(ft_x[:len(t)//2])

period = 1 / freqs[np.argmax(ft_x_abs)]
Amplitude = np.linspace(1,10,100)

plt.figure(figsize=(10,10))
plt.plot(Amplitude, period * np.ones(len(Amplitude)))
plt.grid()
plt.ylabel(r'Time Period')
plt.xlabel(r'Amplitude')
plt.title(r'Anharmonic Oscillator: Time Period vs Amplitude')
plt.savefig('Ques4/4(v).png')
plt.show()

'''
van der Pol oscillator
'''

x0 = [1, 0]
t = np.linspace(0, 20, 100000)
omega, mu = 1, [1, 2, 4, 5]

plt.figure(figsize=(10, 10))

for mu_val in mu:
    soln = RK4(van_der_pol_oscillator, x0, t, omega, mu_val)
    label = '($\omega$, $\mu$) = ({}, {})'.format(omega, mu_val)
    plt.plot(soln[0], soln[1], label=label)

plt.grid()
plt.ylabel('Velocity')
plt.xlabel('Displacement')
plt.title('Van der Pol Oscillator Phase Diagram')
plt.legend(loc='best')
plt.savefig('Ques4/4(vi).png')
plt.show()