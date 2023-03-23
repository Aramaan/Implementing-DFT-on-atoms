import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import gravitational_constant

Vp = 5
R1 = 1
R2 = 4
R3 = 3
R4 = 2
I0 = 3e-6
VT = 5e-2

def J(I):
    U = np.exp((I[3]*R4-I[1]*R2)/VT)
    A2 = (I0*R2*U/VT)
    A4 = (I0*R4*U/VT)
    J = np.array([[0,0,-R3,-R4],
                [-R1,-R2,0,0],
                [0,-A2,1,A4 - 1],
                [1, A2-1,0,-A4]])
    return J

def F(I):
    I5 = I0*(np.exp((I[3]*R4-I[1]*R2)/VT)-1)
    F= np.array([(Vp -R3*I[2] - R4*I[3]),
                 (Vp -R1*I[0] - R2*I[1]),
                 (I[2] +I5 -I[3]),
                 (I[0]-I[1] - I5)])
    return F
    
def Newton(F,J,e):
    Ii = [0,0,0,0]
    If = [1,1,1,1]
    while(np.abs(If[0]-Ii[0])>e or np.abs(If[1]-Ii[1])>e or np.abs(If[2]-Ii[2])>e or np.abs(If[3]-Ii[3])>e ):
        Ii = If
        delI = np.linalg.solve(J(Ii),-F(Ii))
        If = Ii + delI
    return If
I = Newton(F,J,1e-10)
print(I)
print(R1*I[0] - R3*I[2])
