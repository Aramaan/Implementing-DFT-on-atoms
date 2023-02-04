import numpy as np

def gradient(h,dx,dy):
    lx = h.shape[0]
    ly = h.shape[1]
    G = np.zeros((lx,ly,2))
    #central difference for non-edge cases
    for i in range(1,lx-1):
        G[i,:,0] = (h[i+1,:]-h[i-1,:])/(2*dx)
    for j in range(1,ly-1):
        G[:,j,1] = (h[:,j+1]-h[:,j-1])/(2*dy)
    G[0,:,0] = (h[1,:]-h[0,:])/(dx)
    G[-1,:,0] = (h[-1,:]-h[-2,:])/(dx)
    G[:,0,1] = (h[:,1]-h[:,0])/(dy)
    G[:,-1,1] = (h[:,-1]-h[:,-2])/(dy)
    return G



