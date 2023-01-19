'''
Binomial Coefficients
'''
import numpy as np
from math import factorial

'''
10(i)
'''
def bincoeff(n,k):
    return factorial(n)/(factorial(k)*factorial(n-k))

print(bincoeff(5,0))

'''
10(ii)
'''
for i in range(1,21):
    for j in range(0,i+1):
        print(str(bincoeff(i,j)),end=' ')
    print('\n')

'''
10(iii)
'''
