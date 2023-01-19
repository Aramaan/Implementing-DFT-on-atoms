'''
Recursion
'''
'''
12(a)
'''
print('12(a)')
def Catalan(n):
    if n == 0:
        return 1
    else:
        return ((4*n-2)/(n+1))*Catalan(n-1)

print("The 100th catalan number is {}".format(Catalan(100)))
'''
12(b)
'''
print('12(b)')
def GCD(m,n):
    if n==0:
        return m
    else:
        return GCD(n,m%n)

print("The greatest Common Divisor of 108 and 192 is {}".format(GCD(108,192)))
