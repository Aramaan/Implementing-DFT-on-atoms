import numpy as np
from matplotlib import pylot as plt
from scipy.constants import electron_mass,elementary_charge,hbar,epsilon_0
me = electron_mass
e = elementary_charge
e0 = epsilon_0

def grid():


def Vexternal():

def Hartree(n):



def VXC(n):

def numerov(f, x,n, y0, y1, h):
    """
    Solves the differential equation y''(x) = f(x) y(x)
    using the Numerov method.

    Arguments:
    f -- a function of x that returns the value of f(x)
    x0 -- the initial value of x
    xn -- the final value of x
    y0 -- the initial value of y(x0)
    y1 -- the initial value of y(x0 + h)
    h -- the step size

    Returns:
    x -- an array of x values from x0 to xn
    y -- an array of y values corresponding to x
    """

    # Initialize the arrays for x and y
    y = np.zeros(n+1)

    # Set the initial values of y
    y[0] = y0
    y[1] = y1

    # Calculate the coefficients
    coeff = h**2 / 12

    # Use the Numerov method to calculate y for the remaining values of x
    for i in range(1, n):
        y[i+1] = (2*(1-5*coeff*f(x[i]))*y[i] - (1+coeff*f(x[i-1]))*y[i-1]) / (1+coeff*f(x[i+1]))

    return x, y

def SolveKS(V,phi):
    H = V - (1/2)*
    np.linalg.eigh(H)


def edensity(phi):
    n = np.sum((np.absolute(phi))**2,axis=1)
    return n


def Consistency():
