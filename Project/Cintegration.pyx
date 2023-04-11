from array import array
from cpython cimport array

cdef double[:] Numerov(double[:] f, double dh, double[:] x,int size):
    cdef double h2 = dh * dh
    cdef double h12 = h2 / 12.0
    
    cdef double w0 = x[0] * (1 - h12 * f[0])
    cdef double w1 = x[1] * (1 - h12 * f[1])
    cdef double xi = x[1]
    cdef double fi = f[1]
    cdef double w2 = 0.0

    for i in range(2, size):
        w2 = 2 * w1 - w0 + h2 * fi * xi
        fi = f[i]
        xi = w2 / (1 - h12 * fi)
        x[i] = xi
        w0 = w1
        w1 = w2

    return x

cpdef list NumerovF(list f_, double dh, list x_):
    cdef double[:] f = array("d",f_) 
    cdef double[:] x = array("d",x_) 
    cdef double[:] r

    r = Numerov(f,dh,x,len(x))
    return list(r)
