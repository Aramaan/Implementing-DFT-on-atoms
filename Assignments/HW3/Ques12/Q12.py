import numpy as np

A = np.array([[4,1,-1,-1],
              [-1,4,0,-1],
              [-1,0,4,-1],
              [-1,-1,-1,4]])

b = np.array([5,0,5,0])

x = np.linalg.solve(A,b)
print(x)