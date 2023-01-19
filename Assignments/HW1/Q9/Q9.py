'''
The Semi-emprical Mass Formula
'''
'''
9(a)
function to calculate binding energy for given atomic number
 and mass number
'''
def BE(Z,A):
    a1= 15.67
    a2 = 17.23
    a3 = 0.75
    a4 = 93.2
    if A%2 != 0:
        a5 = 0
    else:
        if Z%2 != 0:
            a5 = -12
        else:
            a5 = 12

    BE = a1*A - a2*A**(2/3) -a3*Z**2/A**(1/3) -a4*(A-2*Z)**2/A +a5/A**(1/2)
    return BE

print('The binding energy of an atom with A = 58 and Z = 28 is {} MeV'.format(BE(28,58)))
'''
9(b)
function to calculate binding energy per nucleon for given
 atomic number and mass number
'''
def BEpN(Z,A):
    return BE(Z,A)/A

'''
9(c)
function to calculate the atomic number and binding
energy per nucleon for most stable nuclei for given atomic number
'''
def MSN(Z):
    MS = Z
    BEMS = BEpN(Z,Z)
    for A in range(Z,3*Z):
        N = BEpN(Z,A)
        if N >= BEMS:
            MS = A
            BEMS = N
    return [MS,BEpN(Z,MS)]

'''
9(d)
Finding the most stable nuclei for atominc numbers from 1 to 100
'''
for Z in range(1,101):
    V = MSN(Z)
    print('For atomic number {}, the most stabe nuclei is of atomic number {} with Binding energy per nucleon {}'.format(Z,V[0],V[1]))


