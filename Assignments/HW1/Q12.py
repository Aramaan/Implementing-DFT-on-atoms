'''
Recursion
'''
'''
12(a)
'''
def Catalan(n):
    if n == 0:
        return 1
    else:
        return ((4*n-2)/(n+1))*Catalan(n-1)

print(Catalan(100))
'''
12(b)
'''
def GCD(m,n):
    if n==0:
        return m
    else:
        return GCD(n,m%n)

print(GCD(108,192))
