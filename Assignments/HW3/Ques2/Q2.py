import numpy as np

def f(x):
    return x*(x-1)

def der(f,x,d):
    return ((f(x+d)-f(x))/d)

for i in range(1,8):
    d = 10**(-2*i)
    print('the derivative calculated using delta = {} is {}'.format(d,der(f,1,d)))

'''
The derivative of the function x(x-1) is 2x-2 whose value at x = 1 is 1
the derivative calculated using delta = 0.01 is 1.010000000000001
the derivative calculated using delta = 0.0001 is 1.0000999999998899
the derivative calculated using delta = 1e-06 is 1.0000009999177333
the derivative calculated using delta = 1e-08 is 1.0000000039225287
the derivative calculated using delta = 1e-10 is 1.000000082840371
the derivative calculated using delta = 1e-12 is 1.0000889005833413
the derivative calculated using delta = 1e-14 is 0.9992007221626509
'''    

