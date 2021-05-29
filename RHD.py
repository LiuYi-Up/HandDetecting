
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pickle


hand_connects = frozenset([(0, 16), (13, 14), (14, 15), (15, 16), (0, 4), (1, 2), (2, 3), (3, 4), 
                                                           (0, 8), (5, 6), (6, 7), (7, 8), (0, 12), (9, 10), (10, 11), (11, 12), (0, 20), (17, 18), (18, 19), (19, 20)])

# f = open('/home/lab/Python_pro/Ly_pro/Dataset/HO3D/HO3D_v2/train/GSF12/meta/0120.pkl', 'rb')
f = open('/home/lab/Python_pro/Ly_pro/Dataset/RHD_v1-1/RHD_published_v2/training/anno_training.pickle', 'rb')
a = pickle.load(f)
xyz_label = a[51]
print(xyz_label['uv_vis'])
print(xyz_label)
xs = xyz_label['xyz'][:, 0]
ys = xyz_label['xyz'][:, 1]
zs = xyz_label['xyz'][:, 2]

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
