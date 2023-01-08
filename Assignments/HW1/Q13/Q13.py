#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
Plotting Experimental Data
'''
'''

13(a)
'''
#%%
data = pd.read_csv('Q13/sunspots.txt',delimiter='\t')
data.columns = ['Time','Sunspots']
plt.plot(data['Time'], data['Sunspots'])
plt.title('Number of Sunspots vs Time')
plt.xlabel('Time in months')
plt.ylabel('Number of Sunspots')
plt.savefig("Q13/Q13_a.png")
plt.show()

'''

13(b)
'''
# %%
plt.plot(data['Time'][0:1000], data['Sunspots'][0:1000])
plt.title('Number of Sunspots vs Time')
plt.xlabel('Time in months')
plt.ylabel('Number of Sunspots')
plt.savefig("Q13/Q13_b.png")
plt.show()

'''

13(c)
'''
# %%
Sunspots = data['Sunspots'].rolling(window=11).mean()
plt.plot(data['Time'][0:1000], data['Sunspots'][0:1000])
plt.plot(data['Time'][0:1000], Sunspots[0:1000])
plt.title('Number of Sunspots vs Time')
plt.xlabel('Time in months')
plt.ylabel('Number of Sunspots')
plt.savefig("Q13/Q13_c.png")
plt.show()
# %%
