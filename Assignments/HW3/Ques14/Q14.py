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
    m,n = A.shape
    for k in range(n-1):
        # Compute the pivot element
        pivot = A[k, k]

        # Update the current row of A and b
        for j in range(k+1, min(n, k+3)):
            factor = A[k, j] / pivot
            for i in range(max(0, k-2), k):
                A[j,i] -= factor * A[k, i]
            #b[j] -= factor * b[k]

        # Check for singularity
        if np.abs(A[k+1, k]) < 1e-10:
            raise ValueError("Pivot is too small, the matrix is singular.")

    # Solve the upper-triangular system
    x = np.zeros(n)
    x[n-1] = A[n-1,n-1] / A[n-1, n-2]
    for k in range(n-2, -1, -1):
        x[k] = (b[k] - np.dot(A[k, k+1:min(n, k+2+1)], x[k+1:min(n, k+2+1)])) / A[k, k]
    
    return x

'''
for N = 6
'''
A,w = Init(6)
v1 = np.linalg.solve(A,w)

'''
for N=1e4
'''
#A,w = Init(10000)
#v2 = np.linalg.solve(A,w)

print(v1)
v3 = v1 = banded(A,w)
print(v3)
