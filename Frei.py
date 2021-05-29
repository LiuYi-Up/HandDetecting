import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('/home/lab/Python_pro/Ly_pro/Dataset/FreiHAND_pub_v2/training_xyz.json', 'r') as f:
    a = np.array(json.loads(f.read()))
    print(a[23, :, :])


hand_connects = frozenset([(0, 13), (13, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (3, 4), 
                                                           (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 17), (17, 18), (18, 19), (19, 20)])


xs = a[23, :, 0]
ys = a[23, :, 1]
zs = a[23, :, 2]

 #np.random.rand(n)产生1*n数组，元素大小0-1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for connection in hand_connects:

    start_idx = connection[0]
    end_idx = connection[1]
    x = [xs[start_idx], xs[end_idx]]
    y = [ys[start_idx], ys[end_idx]]
    # z = [abs(hand_landmarks.landmark[start_idx].z), abs(hand_landmarks.landmark[end_idx].z)]
    z = [zs[start_idx], zs[end_idx]]

    ax.plot3D(x, y, z, 'om-')

# ax.scatter(x, y, z)
# n = 100
 
# For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)
 
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
for i in range(len(xs)):
    ax.text(xs[i], ys[i], zs[i], i)
plt.show()
