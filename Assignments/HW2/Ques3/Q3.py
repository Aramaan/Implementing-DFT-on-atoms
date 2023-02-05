
from matplotlib import pyplot as plt
import numpy as np
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.Integration import simpson

def f(t):
    return np.exp(-t**2)

x = np.arange(0,3,0.1)
E = [simpson(0,x[i],10,f) for i in range(len(x))]
#We are using simpson method
plt.plot(x,E)
plt.title('E(x) using simpson method')
plt.xlabel(r'$x$')
plt.ylabel(r'$E(x)$')
plt.savefig('Ques3/Q3.png')
plt.show()



