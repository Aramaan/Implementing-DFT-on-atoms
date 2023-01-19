import numpy as np
import scipy.constants as cons
from astropy.constants import G,R_earth,M_earth
import pandas as pd

'''
Q2(a)
'''
def Altitude():
    T = float(input("Enter the Time period of the satellite:"))
    h = np.cbrt(G.value*M_earth.value/(4*np.pi**2*T**2)) - R_earth.value
    return h
    
if h<=0:
        print("A satellite cannot have that time period")
    else:
        print("The altitude of the satellte is %0.4f meters" %h)


'''
Q2(b)
'''
d = {}
'''
Q2(c)
'''