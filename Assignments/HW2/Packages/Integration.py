from scipy.special import roots_legendre
import numpy as np

'''
Does Trapezoidal integration
 for a given function in certain interval
'''

def trap(a,b,c,d):
    area =((b-a)/2)*(d+c)
    return area

def tIntegraion(x,y):
    integral = [0]
    for i in range(1,len(x)):
        area = trap(x[i-1],x[i],y[i-1],y[i])
        integral.append( integral[i-1] + area)
    return integral

def trapezoidal(a, b, n, f):
    h = float(b - a) / n
    s = 0.0
    s += f(a)/2.0
    for i in range(1, n):
        s += f(a + i*h)
    s += f(b)/2.0
    return s * h

'''
Does Simpson integration
 for a given function in certain interval
'''
def sIntegration(x,y):
    integral = [0]
    s = x[1] -x[0]
    for i in range(2,len(x)):
        if i%2 != 0:
            continue
        area = (s)*(y[i-2]+4*y[i-1]+y[i])/3
        integral.append(integral[i//2-1] + area)
    return integral

def simpson(a, b, N, f):
    s=(b-a)/N
    integral = 0.0
    x= a + s
    for i in range(1,int(N/2) + 1):
        integral += 4*f(x)
        x += 2*s

    x = a + 2*s
    for i in range(1,int(N/2)):
        integral += 2*f(x)
        x += 2*s
    return (s/3)*(f(a)+f(b)+integral)

'''
Does Romberg integration
 for a given function in certain interval
'''

def rombergTRAP(a, b, p, f, I):
    n = 1
    T = np.zeros((p+1,p+1))
    for i in range(1,p+1):
        T[i,1] = I(a,b,n,f)
        for j in range(2,i+1):
            T[i,j] = T[i,j-1]+(T[i,j-1]-T[i-1,j-1])/(4**(j-1)-1) #Richardson Extrp
        n *= 2
    return T[p,p]

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
Does Gauss-Legendre (Gaussiun) Quadrature integration
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
    [y,wy] = mapper(y,wy,a,b)
    x,y = np.meshgrid(x,y)
    wx,wy = np.meshgrid(wx,wy)
    integral = np.sum(wx*wy*f(x,y,*z))
    return integral