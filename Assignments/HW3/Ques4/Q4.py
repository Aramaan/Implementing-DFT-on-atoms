
import numpy as np
import matplotlib.pyplot as plt

def f1(x,y,a,b):
    return y*(a+x**2)

def f2(x,y,a,b):
    return b/(a+x**2)

def FPI_2D(f1,f2,x0,y0,error,max):
    xi = x0
    xf = x0 +1.
    yi = y0
    yf = y0 +1.
    c = 0
    while((np.abs(xf-xi)>error or np.abs(yf-yi)>error) and c<max ):
        c += 1
        xi = xf
        yi = yf
        xf = f1(xi,yi)
        yf = f2(xi,yi)
    return c, xf, yf

max = 1000
a,b,c = FPI_2D(lambda x,y: f1(x,y,1,2),lambda x,y:f2(x,y,1,2),0.5,0.5,1e-6,max)
if (a == max):
        print("The Iteration didn't converge")
else:
     print("for a = 1 and b = 2 the solution of the equation that converged in {} steps \n is x = {}, y = {} with accuracy of {}".format(a,b,c,1e-6))

     

def f1(x,y,a,b):
    return np.sqrt(b/y-a)

def f2(x,y,a,b):
    return x/(a+x**2)

a,b,c = FPI_2D(lambda x,y: f1(x,y,1,2),lambda x,y:f2(x,y,1,2),0.5,0.5,1e-6,max)
if (a == max):
        print("The Iteration didn't converge")
else:
     print("for a = 1 and b = 2 the solution of the equation that converged in {} steps \n is x = {}, y = {} with accuracy of {}".format(a,b,c,1e-6))

'''
The Iteration didn't converge
for a = 1 and b = 2 the solution of the equation that converged in 26 steps 
is x = 1.9999986844792996, y = 0.40000023873260093 with accuracy of 1e-06
'''