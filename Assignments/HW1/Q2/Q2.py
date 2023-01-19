import numpy as np
import scipy.constants as cons
from astropy.constants import G,R_earth,M_earth
import pandas as pd

'''
Q2(a)
'''
T = float(input("Enter the Time period of the satellite in seconds:"))
h = np.cbrt(G.value*M_earth.value//4/np.pi**2*T**2) - R_earth.value
    
if (h <= 0):
    print("A satellite cannot have that time period")
else:
    print("The altitude of the satellte is %0.4f meters" %h)



"""
 Q2 (b) 
1 day       : Altitude of the satellite is 35862994.1977 m
90 minutes  : Altitude of the satellite is 274455.4688 m
45 minutes  : A satellite cannot have that time period

Since less height implies lower time period, the height above the earth's surface
must be positive leading to lower bound to the time period due to earth's surface.
 Thus a satellite with 45 minutes time period is impossible.
"""

""" 
 Q2 (c) 
A sidereal day is the time taken by the earth to rotate 360 deg around
 its axis with respect to the star on the background which takes 23.94 hours.
 However a solar day is the time taken by the earth to rotate 360 deg around its axis with respect to the sun. 
 This is equal to 24 hours. A solar day is not a measure of true time period of
rotation of earth as it incorporates the revolution of earth around the sun as well. 
A sidereal day is the true time it takes for the earth to complete one rotation.



For 1 sidereal day, height is 35780818.7579 m
For 1 solar day, height is 35862994.1977 m


Height difference: 82175.4397 m
For geo syncrounous orbit, Satellite must be placed this much lower in height.
"""