import time
'''
Prime Numbers
'''
s = time.time()
import numpy as np
primes = [2]

'''
checks whether the input value is prime by
 checking divisibility with all the values in the array 'primes'

 input:
 number to be checked

 output:
 true or False
'''
def isprime(n):
    s = np.sqrt(n)
    for i in primes:
        if n%i == 0:
            return False
        if i >= s:
            return True
    

for i in range(3,10000):
    if isprime(i):
        primes.append(i)
    else:
        continue

print(primes)
print(time.time()- s)
