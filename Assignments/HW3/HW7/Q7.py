import numpy as np
from matplotlib import pyplot as plt


def P(x):
    return 924*x**6 - 2772*x**5 +3150*x**4 - 1680*x**3 + 420*x**2 -42*x +1

def derP(x):
    h = 1e-4
    return (P(x+h)-P(x-h))/(2*h)
    
def Newton(xi,e):
    x = xi
    w = 0
    while (np.abs(w-x)>e):
        w = x
        x = w - P(w)/derP(w)
    return x

y = np.linspace(0,1,100)
PP = P(y)
plt.plot(y,PP)
plt.grid()
plt.show()


'''
The approximate values of the roots of the above equation are
x = [0.05, 0.15, 0.35, 0.65, 0.85, 0.95]
Using these value we calcululate the exact values using newton's method
'''

x = [0.05, 0.15, 0.35, 0.65, 0.85, 0.95]
y = []
e = 1e-6
for i in x:
    y.append(Newton(i,e))
print(y)

'''
The exact roots of the equation are
x(exact) = [0.03376524289842393, 0.16939530676688427, 0.3806904069582371, 0.6193095930417563, 0.8306046932330663, 0.9662347571015868]
'''


