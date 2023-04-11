import numpy as np
import matplotlib.pyplot as plt

def k(x):
    """
    Example k(x) function
    """
    return np.repeat(-1,len(x))

def analytical_solution(x):
    """
    Analytical solution for the ODE
    """
    return np.exp(-x)

def numerov_nonuniform(k, x, y0, y1):
    """
    Numerov algorithm for solving second-order ODE with non-uniform grid.

    Parameters
    ----------
    x : array-like
        Grid points. Must be a 1D array of shape (N,).
    y0 : float
        Initial value of y at x[0].
    y1 : float
        Initial value of y at x[1].
    k : array-like
        Function k(x) for the ODE. Must be a 1D array of shape (N,).

    Returns
    -------
    y : array
        Numerical solution of y(x) on the grid x.
    """
    N = len(x)
    h = np.diff(x)
    y = np.empty(N)
    y[0] = y0
    y[1] = y1

    for i in range(2, N):
        h1 = h[i-1]
        h2 = h[i-2]
        xi = x[i]
        xi_1 = x[i-1]
        xi_2 = x[i-2]
        ki = k[i]
        ki_1 = k[i-1]
        ki_2 = k[i-2]
        fi = 12 * (h1**2) * ki

        w1 = 1 + (1/12) * (h1**2) * ki_1
        w2 = 1 + (1/12) * (h2**2) * ki_2
        w3 = 1 + (1/12) * (h1**2) * ki

        y[i] = (2*w1*y[i-1] - w2*y[i-2] + h1**2 * fi * y[i]) / w3

    return y

# Example usage
# Define x values and step sizes h
x = np.linspace(-5, 5, num=100)  # Example x values
h = np.diff(x)  # Step sizes between x values

# Define initial conditions
y0 = analytical_solution(x[0])  # y(x0) using analytical solution
y1 = analytical_solution(x[1])  # y(x0 + h0) using analytical solution

# Solve the ODE using Numerov algorithm with non-uniform grid
y_numerov = numerov_nonuniform(k(x), x, y0, y1)
print(y_numerov)
# Calculate the analytical solution for comparison
y_analytical = analytical_solution(x)
print(y_analytical)
#Plot the numerical and analytical solutions
plt.plot(x, y_numerov, label='Numerical solution')
# plt.ylim(0,150)
plt.show()
plt.plot(x, y_analytical, label='Analytical solution')
# plt.ylim(0,150)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical vs. Analytical solution')
plt.legend()
plt.show()

'''
In this example, we define an analytical solution for the ODE using the analytical_solution function, and then use this solution to set the initial conditions y0 and y1 for the numerov_solve_nonuniform function. We then compare the numerical solution obtained from numerov_solve_nonuniform with the analytical solution by plotting both on the same graph using matplotlib. If the numerical solution closely matches the analytical solution, it indicates that the numerov_solve_nonuniform function is working correctly.

'''