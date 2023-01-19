'''
Binomial Coefficients
'''
import numpy as np
from math import factorial

'''
10(i)
'''
print('10(i)')
def bincoeff(n,k):
    return factorial(n)/(factorial(k)*factorial(n-k))

print(bincoeff(5,0))

'''
10(ii)
'''
print('10(ii)')
for i in range(1,21):
    for j in range(0,i+1):
        print(str(bincoeff(i,j)),end=' ')
    print('\n')

'''
10(iii)
'''
print('10(iii)')
# part (a)
print("Probability of getting 60 heads out of 100 tosses is %.5f"%(bincoeff(100, 60)/2**100))

#part (b)
list_coeff = [bincoeff(100,k) for k in range(60,101)]
print("Probability of getting 60 or more heads in 100 tosses is %.5f"%(sum(list_coeff)/2**100))
