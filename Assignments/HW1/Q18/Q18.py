
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.constants import h

data = pd.read_csv('Q18/millikan.txt',delimiter=' ')


def Average(a,N):
    return np.sum(a)/N

def crAverage(a,b,N):
    return np.sum(np.dot(a,b))/N


def line(x,y):
    N = len(x)
    m = (crAverage(x,y,N)-Average(x,N)*Average(y,N))/(crAverage(x,x,N)-Average(x,N)*Average(x,N))
    c = (crAverage(x,x,N)*Average(y,N)-Average(x,N)*crAverage(x,y,N))/(crAverage(x,x,N)-Average(x,N)*Average(x,N))
    return [m,c]

x = data.iloc[:,0]
y = data.iloc[:,1]
plt.scatter(x,y)
plt.title('Milikan')
plt.xlabel("Frequency of light (Hz)")
plt.ylabel("Stopping Voltage (V)")
m,c = line(x,y)
#lt.plot(x,l[0]*x + l[1],'g-.')

Y = m * x + c

e = 1.602e-19
h_exp = m * e
print("Planck's constant from data = %.2E \n Standard value = %.2E \n Percentage error = %.2f" % (
h_exp, h, (h - h_exp) / h * 100))
plt.figure()
plt.plot(x, y, 'k.')
plt.plot(x, Y, '--')
plt.text(0.55e15,2, "Planck's constant from data = %.2E \n Standard value = %.2E \n Percentage error = %.2f" % (
h_exp, h, (h - h_exp) / h * 100), fontsize = 10)
plt.title('Milikan')
plt.xlabel("Frequency of light (Hz)")
plt.ylabel("Stopping Voltage (V)")
plt.savefig("Q18/HW1_Q18.png")
plt.show()


