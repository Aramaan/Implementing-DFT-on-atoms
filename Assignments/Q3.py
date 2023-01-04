import numpy as np

x = float(input("input the x coordinate:"))
y = float(input("input the x coordinate:"))

r = np.sqrt(x**2 + y**2)
if x>0 and y>0:
    t = np.arctan(y/x)*180/np.pi
elif x==0 and y>0:
    t = (np.pi)*180/np.pi
elif x<0 and y>=0:
    t = (np.pi + np.arctan(y/x))*180/np.pi
elif x==0 and y<0:
    t = -(np.pi)*180/np.pi
elif x<0 and y<0:
    t = (np.pi + np.arctan(y/x))*180/np.pi
else:
    t = (2*np.pi + np.arctan(y/x))*180/np.pi


print("The polar coordinates of (x,y) = (%0.4f,%0.4f) are (radius,angle) = (%0.4f,%0.4f deg)" % (x,y,r,t))