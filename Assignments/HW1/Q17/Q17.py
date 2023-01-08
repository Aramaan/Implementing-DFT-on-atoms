#%%
import numpy as np
from matplotlib import pyplot as plt

# %%

def func(z,c):
    return np.square(z)+c

def inMset(c,n):
    z = 0
    for i in range(n):
        z = func(z,c)
        if (np.absolute(z) > 2):
            return i
    return n

def grid(N):
    A =np.zeros((N,N),dtype=np.complex_)
    for k in range(N):
        for i in range(N):
            A[k,i] = complex((-2 + 4*i/N) + (-2j + 4j*k/N))
    return A
    
# %%
N = 1000
n = 100
A = grid(N)
B = np.zeros((N,N))
for i in range(N):
        for k in range(N):
            B[k,i] = inMset(A[i,k],n)

B = np.transpose(B)
B

# %%
plt.imshow(B,cmap = 'magma')
plt.colorbar()
plt.savefig('Q17/Mset.png')
plt.show()

# %%
np.square(3+2j)
# %%
