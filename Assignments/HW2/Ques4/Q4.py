#%%
from matplotlib import pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.abspath(".."))
from Ques2.Q2 import sIntegration

# %%
def J(theta,m,x):
    j = np.cos(m*theta-x*np.sin(theta))
    return j



# %%
