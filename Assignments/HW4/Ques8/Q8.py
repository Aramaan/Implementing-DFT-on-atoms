
import numpy as np
import matplotlib.pyplot as plt

photo = np.loadtxt('Ques8/blur.txt')
sy,sx = photo.shape
dev = 25

x = np.arange(0,sy,1,dtype=np.float32)
y = np.arange(0,sy,1,dtype=np.float32)

plt.figure(figsize=(10,10))
plt.pcolormesh(x,-y,photo)
plt.savefig('Ques8/8(i).png')
plt.show()

fy = np.zeros([sy,])
fx = np.zeros([sx,])

for i in range(sy):
    if y[i] < sy/2:
        fy[i] = np.exp(-(y[i])**2/(2*dev**2))
    else:
        fy[i] = np.exp(-(y[i]-sy)**2/(2*dev**2))
 #   if x[i] < sy/2:
 #       fx[i] = np.exp(-(x[i])**2/(2*dev**2))
  #  else:
  #      fx[i] = np.exp(-(x[i]-sx)**2//(2*dev**2))

fx = fy

fx, fy = np.meshgrid(fx,fy)

f = fx*fy

plt.figure(figsize=(10,10))
plt.pcolormesh(x,-y,f,cmap='gray')
plt.savefig('Ques8/8(ii).png')
plt.show()

fft_f = np.fft.rfft2(f)
fft_photo = np.fft.rfft2(photo)

# to avoid division by 0
e = 1e-3
fft_f = np.piecewise(fft_f,[np.abs(fft_f)<=e],[e,lambda x: x])

fft_photoN = fft_photo/fft_f
photoN = np.fft.irfft2(fft_photoN)

plt.figure(figsize=(10,10))
plt.pcolormesh(x,-y,photoN,cmap='gray')
plt.savefig('Ques8/8(iii).png')
plt.show()

