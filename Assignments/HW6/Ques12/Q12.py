import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import G
G =1
def RK4(ode_func, y0, t):
    """
    Implementation of the Fourth Order Runge-Kutta (RK4) method.
    Args:
        ode_func (function): ODE function f(t, y) to be solved.
        y0 (float): Initial value of the dependent variable.
        t0 (float): Initial value of the independent variable.
        t_end (float): End value of the independent variable.
        h (float): Step size.
    Returns:
        tuple: Tuple containing two arrays: t (array of time steps) and y (array of solutions).
    """
    h = t[1]-t[0]
    n = len(t)
    y = np.zeros([len(y0),n+1])
    y[:,0] = y0

    for i in range(n):
        k1 = h * ode_func(t[i], y[:,i])
        k2 = h * ode_func(t[i] + h/2, y[:,i] + k1[:,0]/2)
        k3 = h * ode_func(t[i] + h/2, y[:,i] + k2[:,0]/2)
        k4 = h * ode_func(t[i] + h, y[:,i] + k3[:,0])

        y[:,i + 1] = y[:,i] + (k1[:,0] + 2*k2[:,0] + 2*k3[:,0] + k4[:,0]) / 6

    return y[:,:n]

# def RK4(f,x0,t,*args): #f must be a matrix of order (dim,1)
#     if isinstance(t,list): 
#         t = np.array(t)
#     elif not(isinstance(t,np.ndarray) or isinstance(t,list)): 
#         t = np.array([0,t])
        
#     if isinstance(x0,list): 
#         x0 = np.array(x0)
#     elif not(isinstance(x0,np.ndarray) or isinstance(x0,list)): 
#         x0 = np.array([x0])

#     x = np.zeros((x0.shape[0],t.shape[0],),dtype=np.float64)
#     h = t[1] - t[0]
#     x[:,0] = x0 #Initial Condition
#     dim = x0.shape[0]
    
#     for i in range(1,t.shape[0]):
#         k1, k2, k3, k4 = None, None, None, None
        
#         if (dim==1):
#             k1 = f(t[i-1],x[:,i-1],*args)*h
#             k2 = f(t[i-1]+h/2,x[:,i-1]+k1/2,*args)*h
#             k3 = f(t[i-1]+h/2,x[:,i-1]+k2/2,*args)*h
#             k4 = f(t[i-1]+h,x[:,i-1]+k3,*args)*h
            
#             x[:,i] = x[:,i-1] + (k1+2*(k2+k3)+k4)/6
#         else:
#             k1 = f(t[i-1],x[:,i-1],*args)*h
#             k2 = f(t[i-1]+h/2,x[:,i-1]+k1[:,0]/2,*args)*h
#             k3 = f(t[i-1]+h/2,x[:,i-1]+k2[:,0]/2,*args)*h
#             k4 = f(t[i-1]+h,x[:,i-1]+k3[:,0],*args)*h
            
#             x[:,i] = x[:,i-1] + (k1[:,0]+2*(k2[:,0]+k3[:,0])+k4[:,0])/6
    
#     if (x.shape[0]==1): return x[0]
#     return x

def adaptive_RK4(ode_func, y0, tu, tol):
    """
    Implementation of the adaptive Runge-Kutta method for a single ODE.
    Args:
        ode_func (function): ODE function F(t, y) to be solved.
        y0 (float): Initial value of the dependent variable.
        t0 (float): Initial value of the independent variable.
        t_end (float): End value of the independent variable.
        tol (float): Tolerance for adaptive step size.
    Returns:
        tuple: Tuple containing two arrays: t (array of t steps) and y (array of solutions).
    """
    
    x, t = [y0], [tu[0]]
    h = tu[1] - tu[0]
    t_end = tu[-1]
    c = 0
    print(c)
    print(t_end)
    # err = np.abs(y2 - y1) / 15
    # # Compute scaling factor
    # scale = tol / (np.max(err) + 1e-8)
    # h *= np.sqrt(scale)  # Adjust step size based on scaling factor
    while (t[c]<=t_end):
        print(c)
        c += 1
        x1 = RK4(ode_func,x[c-1],[t[c-1],t[c-1]+h,t[c-1]+2*h])[:,-1]            
        x2 = RK4(ode_func,x[c-1],[t[c-1],t[c-1]+2*h])[:,-1]
        print(x1)
        x.append(x1)
        t.append(t[c-1]+h)
        print(t[c])
        err = np.abs(x2 - x1) / 15
         # Compute scaling factor
        scale = tol / (np.max(err) + 1e-7)
        h *= np.sqrt(scale)  # Adjust step size based on scaling factor
        # diff = np.sqrt(np.abs(np.dot(x1-x2,(x1-x2).T)))
        # print(diff)
        # h = h * (10*tol/diff)**(1/2)
        print(t[c]<=t_end)
    
    x = np.array(x).T
    t = np.array(t)
    if (x.shape[0]==1): return x[0]
    return t,x

def f(t,x,M): #F is an array of functions of time
    dim = len(x)//(2*len(M))
    if (len(x)%(2*len(M))!=0): raise Exception('Improper Intial Conditions!')        
    pos = np.array([x[i] for i in range(0,len(x),2)],dtype=np.float32)
    vel = np.array([x[i] for i in range(1,len(x),2)],dtype=np.float32)
    acc = np.zeros(len(pos),dtype=np.float32)
    for i in range(0,len(pos)):
        for j in range(0,len(pos)):
            if (j//dim!=i//dim and j%dim==i%dim): #different body and same coordinates
                dist = 0
                for k in range((j//dim)*dim,(j//dim+1)*dim):
                    dist += (pos[k]-pos[i])**2
                dist = np.sqrt(dist)               
                acc[i] += G*M[j//dim]*(pos[j]-pos[i])/dist**3

    vector = np.zeros((2*len(pos),1))
    counter = 0
    for i in range(0,2*len(pos),2):
        vector[i] = vel[counter]
        vector[i+1] = acc[counter]
        counter += 1
    return vector

def func(t,x,m): #F is an array of functions of t
    x1,vx1,y1,vy1,x2,vx2,y2,vy2,x3,vx3,y3,vy3 = x
    m1,m2,m3 = m[0],m[1],m[2]
    r21 = np.sqrt(np.square(x2-x1) +  np.square(y2-y1))
    r32 = np.sqrt(np.square(x3-x2) +  np.square(y3-y2))
    r13 = np.sqrt(np.square(x1-x3) +  np.square(y1-y3))
    dx1dt = vx1
    dvx1dt = G*m2*(x2-x1)/r21**3 + G*m3*(x3-x1)/r13**3
    dy1dt = vy1
    dvy1dt = G*m2*(x2-x1)/r21**3 + G*m3*(x3-x1)/r13**3
    dx2dt = vx2
    dvx2dt = G*m3*(x3-x2)/r32**3 + G*m1*(x1-x2)/r21**3
    dy2dt = vy2
    dvy2dt = G*m3*(x3-x2)/r32**3 + G*m1*(x1-x2)/r21**3
    dx3dt = vx3
    dvx3dt = G*m1*(x1-x3)/r13**3 + G*m2*(x2-x3)/r32**3
    dy3dt = vy3
    dvy3dt = G*m1*(x1-x3)/r13**3 + G*m2*(x2-x3)/r32**3
    return np.array([[dx1dt,dvx1dt,dy1dt,dvy1dt,dx2dt,dvx2dt,dy2dt,dvy2dt,dx3dt,dvx3dt,dy3dt,dvy3dt]]).T



t = np.linspace(0,2,2000) #t in code units
z0 = [3,0,1,0,-1,0,2,0,-1,0,1,0] # xjb vjb
#z0 = np.random.randint(-5,5,size = 12)
print(z0)
m = [150,200,250]
tol = 1e-6
r = RK4(lambda t,y: func(t,y,m),z0,t)
plt.figure(figsize=(13,10))
# plt.plot(t)
plt.plot(r[0],r[2],label='Mass No. %d'%1)
plt.plot(r[2],r[6],label='Mass No.%d'%2)
plt.plot(r[8],r[10],label='Mass No. %d'%3)
plt.grid()
plt.ylabel(r'$y$')
plt.xlabel(r'$x$')
plt.title(r'Trajectory of the particles')
plt.legend(loc='best')
plt.savefig('Ques12/12.png')
plt.show()
