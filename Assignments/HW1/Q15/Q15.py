'''
STM measurement of (111) Silicon
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("Q15/stm.txt", header=None, delimiter=' ')
s = data.shape
X, Y = np.meshgrid(range(s[1]), range(s[0]))

'''
Heat map 
'''

plt.figure(1)
plt.pcolormesh(data,cmap = 'magma')
plt.colorbar()
plt.title("STM measurement of (111) Silicon (Heat Map)")
plt.savefig('Q15heatmap.png')
plt.show()

'''
Surface plot
'''
plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, data ,cmap='magma', edgecolor='none')
ax.set_title('STM measurement of (111) Silicon (Surface plot)')
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig('Q15surfaceplot.png')
plt.show()  