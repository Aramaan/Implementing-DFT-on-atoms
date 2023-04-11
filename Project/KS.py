from scipy import *
from scipy import integrate
from scipy import optimize
from matplotlib import pyplot as plt
import numpy as np
#from Cintegration import NumerovF

def KS(V,R):

    def Numerovc(f, x0_, dx, dh_):
        """
        Numerovc function for solving a second-order ODE
        Args:
            f (array): Array of function values f(x)
            x0_ (float): Initial x value
            dx (float): Step size for x
            dh_ (float): Step size for h
        Returns:
            x (array): Array of x values
        """
        h2 = dh_**2
        h12 = h2/12.

        x = np.zeros(len(f))
        x[0] = x0_
        x[1] = x0_ + dh_*dx
        xi = x[1]
        fi = f[1]
        
        for i in range(2, len(f)):
            w2 = 2*x[i-1] - x[i-2] + h2*fi*xi
            fi = f[i]
            xi = w2/(1 - h12*fi)
            x[i] = xi

        return x
    
    # def Numerovc(f, x0_, dx, dh_):
    #     x = np.zeros(len(f))
    #     dh=float(dh_)
    #     x[0]=x0_
    #     x[1]=x0_+dh*dx
    #     x = Cintegration.Numerov(x,dh,f)
    #     return x


    def fSchrod(En, l, R):
        #V = -1/R
        return l*(l+1.)/R**2+2*V-2*En

    def ComputeSchrod(En,R,l):
        "Computes Schrod Eq." 
        f = fSchrod(En,l,R[::-1])
        ur = Numerovc(f,0.0,-1e-7,-R[1]+R[0])[::-1]
        norm = integrate.simps(ur**2,x=R)
        return ur*1/np.sqrt(abs(norm))

    def Shoot(En,R,l):
        ur = ComputeSchrod(En,R,l)
        ur = ur/R**l
        f0 = ur[0]
        f1 = ur[1]
        f_at_0 = f0 + (f1-f0)*(0.0-R[0])/(R[1]-R[0])
        return f_at_0

    def FindBoundStates(R,l,nmax,Esearch):
        n=0
        Ebnd=[]
        u0 = Shoot(Esearch[0],R,l)
        for i in range(1,len(Esearch)):
            u1 = Shoot(Esearch[i],R,l)
            if u0*u1<0:
                Ebound = optimize.brentq(Shoot,Esearch[i-1],Esearch[i],xtol=1e-16,args=(R,l))
                Ebnd.append((l,Ebound))
                if len(Ebnd)>nmax: break
                n+=1
                print('Found bound state at E=%14.9f E_exact=%14.9f l=%d' % (Ebound, -1.0/(n+l)**2,l))
            u0=u1
    
        return Ebnd
        
    Esearch = -1.2/np.arange(1,20,0.2)**2

    #R = np.linspace(1e-8,100,2000)

    nmax=7
    Bnd=[]
    for l in range(nmax-1):
        print(l)
        Bnd += FindBoundStates(R,l,nmax-l,Esearch)
        
    Bnd.sort(key=lambda x: (x[1], x[0]))

    Z=19  # Like Ni ion

    N=0
    u = []
    rho= np.zeros(len(R))
    L, E = [],[]
    for (l,En) in Bnd:
        #ur = SolveSchroedinger(En,l,R)
        L.append(l)
        E.append(En)
        ur = ComputeSchrod(En,R,l)
        u.append(ur)
        dN = 2*(2*l+1)
        if N+dN<=Z:
            ferm=1.
        else:
            ferm=(Z-N)/float(dN)
        drho = ur**2 * ferm * dN/(4*np.pi*R**2)
        rho += drho
        N += dN
        print('adding state', (l,En), 'with fermi=', ferm)
        plt.plot(R, drho*(4*np.pi*R**2))
        if N>=Z: break
    plt.xlim([0,25])
    plt.show()
    plt.plot(R,rho*(4*np.pi*R**2),label='charge density')
    plt.xlim([0,25])
    plt.show()
    u = np.array(u)
    print(L)
    print(E)
    print(u)
    return L,E,u