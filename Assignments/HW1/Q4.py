'''
Special Relativity
'''
import numpy as np  

'''
4(a)
'''

def TimeSpaceShip(x,v):
    T = x*np.sqrt(1-(v)**2)/v
    return T

def TimeEarth(x,v):
    T = x/v
    return T


x = float(input("Enter the distance x in light years: "))
v = float(input("Enter the speed v as a fraction of speed of light c: "))

print("Time taken in the rest frame on Earth is %.4f years" % TimeEarth(x,v))
print("Time taken  as per by a passenger on the space ship is %.4f years"% TimeSpaceShip(x,v))

'''
4(b)
If we assume the distance to be 10 light years and the spaceship
to be moving with 99% of the speed of light,

Time taken in the rest frame on Earth is 10.1010 years and
Time taken  as per by a passenger on the space ship is 1.4249 years
'''

