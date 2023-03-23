import numpy as np
import pandas as pd

def Init(N):
    A = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            if (i == j):
                if(i==N-1 and j==N-1) or ((i==0 and j==0)):
                    A[i,j]=3
                else:
                    A[i,j]=4
            elif(np.abs(i-j)<=2):
                A[i,j]=-1
            else:
                A[i,j]=0
    w = np.zeros([N,1])
    w[0] = 5
    w[1] = 5
    return A,w

import numpy as np

def banded(A, b):
    """
    Perform banded Gaussian elimination on the system Ax = b
    """
    n = A.shape[0]
    a = np.concatenate((A,b),axis=1)

    # Applying Gauss Elimination
    for i in range(n):
        pivot =  a[i][i]
        for j in range(i+1, min(n,i+3)):
            ratio = a[j][i]/pivot
            for k in range(0,n+1):
                a[j][k] = a[j][k] - ratio * a[i][k]

    # Back Substitution
    x = np.zeros(n)
    x[n-1] = a[n-1][n]/a[n-1][n-1]

    for i in range(n-2,-1,-1):
        
        x[i] = a[i][n]
        
        for j in range(i+1,n):
            x[i] = x[i] - a[i][j]*x[j]
        
        x[i] = x[i]/a[i][i]

    return x

'''
for N = 6
'''
A,w = Init(6)
v1 = banded(A,w)
print(v1)
'''
for N=1e4
'''
A,w = Init(10000)
v2 = banded(A,w)
np.savetxt(r"Ques14/potentials.txt",v2)

