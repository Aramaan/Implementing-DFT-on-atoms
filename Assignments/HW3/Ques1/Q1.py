import numpy as np

def quad(a,b,c):
    det = np.sqrt(b**2-4*a*c)
    return [(-b+det)/(2*a),(-b-det)/(2*a)]
print(quad(0.001,1000,0.001))

def quad2(a,b,c):
    det = np.sqrt(b**2-4*a*c)
    return [(2*c)/(-b-det),(2*c)/(-b+det)]
print(quad2(0.001,1000,0.001))