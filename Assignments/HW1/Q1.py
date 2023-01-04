import numpy as np
import scipy.constants as con

h = float(input("Enter the height: "))


print("Time taken by the ball to hit the ground: %.4f seconds" % np.sqrt(2*h/con.g))

