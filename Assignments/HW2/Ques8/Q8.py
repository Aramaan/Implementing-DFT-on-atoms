import numpy as np
import sys, os.path
sys.path.append(os.path.abspath('.'))
from  Packages.Integration import simpson

def f(x):
    return (np.sin(np.sqrt(100*x)))**2