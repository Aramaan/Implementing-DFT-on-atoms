import numpy as np

A = np.array([[4,1,-1,-1],
              [-1,4,0,-1],
              [-1,0,4,-1],
              [-1,-1,-1,4]])

b = np.transpose(np.array([5,0,5,0]))

def gaussian(A, b):
    """
    Solves a system of linear equations using Gaussian elimination.
    """

    n = A.shape[0]
    A = A.copy()
    b = b.copy()


    for k in range(n-1):
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k+1:] = A[i, k+1:] - factor * A[k, k+1:]
            b[i] = b[i] - factor * b[k] 

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x

x = gaussian(A,b)
print(np.matmul(A,x))

