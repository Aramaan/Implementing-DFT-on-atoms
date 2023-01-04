'''
Prime Numbers
'''
import numpy as np
primes = [2]

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