from scipy.special import roots_legendre
import numpy as np

'''
maps the points and weights for 
integration in any interval [a,b] to the interval
[-1,1]
'''
def mapper(x,w,a,b):
    x = ((b+a)/2 + ((b-a)/2)*x)
    w = ((b-a)/2)*w
    return [x,w]

'''
Does Gauss-Legendre (Gaussiun) Quadrature interation
 for a given function in certain interval
'''
def GaussQuad(a,b,N,f):
    [x,w] = roots_legendre(N+1)
    [x,w] = mapper(x,w,a,b)
    integral = np.sum(w*f(x))
    return integral

def GaussQuad2d(a,b,n,f,*z):
    [x,wx] = roots_legendre(n+1)
    [y,wy] = roots_legendre(n+1)
    [x,wx] = mapper(x,wx,a,b)
    x,y = np.meshgrid(x,y)
    wx,wy = np.meshgrid(wx,wy)
    integral = np.sum(wx*wy*f(x,y,*z))
    return integral