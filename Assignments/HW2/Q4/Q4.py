#%%
from matplotlib import pyplot as plt
import numpy as np
from Ques2.Q2 import sIntegration

# %%
def J(theta,m,x):
    j = np.cos(m*theta-x*np.sin(theta))
    return j



# %%
