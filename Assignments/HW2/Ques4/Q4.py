#%%
from matplotlib import pyplot as plt
import numpy as np
import Ques2.Q2

# %%
def J(theta,m,x):
    j = np.cos(m*theta-x*np.sin(theta))
    return j



# %%
