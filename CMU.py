import json


with open('/home/lab/Python_pro/Ly_pro/Dataset/CMU_HAND/hand_labels/manual_test/002058449_01_l.json', 'r') as f:
    a = json.loads(f.read())
    print(a)