'''
Catalan numbers
'''
print('catalan numbers')
def successor(c,i):
    return ((4*i+2)/(i+2))*c

i=0
C=1
while C <= 1000000:
    print(C)
    C = successor(C,i)
    i += 1