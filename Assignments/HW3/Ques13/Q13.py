import numpy as np

R1,R3,R5 = 1e3,1e3,1e3
R2,R4,R6 = 2e3,2e3,2e3
C1 = 1e-6
C2 = 5e-7
x = 3
w = 1e3
A = np.array([[1/R1 + 1/R4 + w*C1*1j,-w*C2*1j,0],
              [-w*C1*1j,1/R2+1/R5+w*C1*1j+w*C2*1j,-w*C2*1j],
              [0,-w*C2*1j,1/R3+1/R6+w*C2*1j]],dtype=complex)
b = [x/R1,x/R2,x/R3]
V = np.linalg.solve(A,b)
v = np.absolute(V)
phi = np.angle(V)
print(v,phi)