'''
Planetary Orbits
'''

import numpy as np
from astropy.constants import G, M_sun

l1 = float(input("Perihelion distance: "))
v1 = float(input("Perihelion velocity: "))

a = 1/(2/l1 - v1**2/G.value/M_sun.value)
e = 1 - l1/a
l2 = a*(1+e)
v2 = l1*v1/l2
T = np.sqrt(4*np.pi**2/G.value/M_sun.value*a**3)/3.15576e7

print("Aphelion distance: %.4f" % l2)
print("Aphelion velocity: %.4f" % v2)
print("Orbital period: %.4f" % T)
print("Eccentricity: %.4f" % e)

'''
Earth :-
Perihelion distance - 1.4710e11
Perihelion velocity- 3.0287e4
Aphelion distance- 152111350728.5926
Aphelion velocity- 29289.1864
Orbital period- 1.0001
Eccentricity- 0.0167

Halley's Comet :-
Perihelion distance- 8.7830e10
Perihelion velocity- 5.4529e4
Aphelion distance- 5371566481143.3535
Aphelion velocity- 891.5988
Orbital period- 77.9457
Eccentricity- 0.9678
'''