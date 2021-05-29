import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


data = sio.loadmat('/home/lab/Python_pro/Ly_pro/Dataset/STB/labels/B4Counting_SK.mat')
result = data['handPara']
print(result[:,:,71])


hand_connects = frozenset([(0, 13), (13, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (3, 4), 
                                                           (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 17), (17, 18), (18, 19), (19, 20)])


xs = result[0, :, 171]
ys = result[1, :, 171]
zs = result[2, :, 171]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for connection in hand_connects:

    start_idx = connection[0]
    end_idx = connection[1]
    x = [xs[start_idx], xs[end_idx]]
    y = [ys[start_idx], ys[end_idx]]
    z = [zs[start_idx], zs[end_idx]]

    ax.plot3D(x, y, z, 'om-')
# ax.scatter(xs, ys, zs)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
for i in range(len(xs)):
    ax.text(xs[i], ys[i], zs[i], i)
plt.show()
