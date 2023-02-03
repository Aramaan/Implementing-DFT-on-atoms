import numpy as np
from matplotlib import pyplot as plt

'''
Gamma Function
'''

def Integrand(a,x):
    return x**(a-1)*np.exp(-x)

x = np.linspace(0,5,100)


for a in range(2,5):
    y = Integrand(a,x)
    plt.plot(x,y)
plt.show()

