import pickle


f = open('/home/lab/Python_pro/Ly_pro/Dataset/HO3D/HO3D_v2/train/GSF12/meta/0620.pkl', 'rb')
# f = open('/home/lab/Python_pro/Ly_pro/Dataset/HO3D/HO3D_v2/evaluation/AP10/meta/0000.pkl', 'rb')
a = pickle.load(f)
# print('handJoints3D:\n{}'.format(a['handJoints3D'][0]))
for key, val in a.items():  # a.item() 返回可遍历的键，值元组
    print('{}:\n{}' .format(key, val))


    