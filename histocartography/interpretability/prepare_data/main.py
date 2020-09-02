import numpy as np
import os
import glob
from shutil import copy2

tumor_types = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']

read_path = '/Users/pus/Desktop/Projects/Data/Histocartography/BRACS_L/Images_norm/'
save_path = '/Users/pus/Desktop/Projects/Data/Histocartography/explainability/Images_norm/'

test_ids = [291, 735, 737, 738, 739, 1232, 1284, 1287, 1288, 1228, 1241, 1271, 1283, 1296, 1321, 1326, 1327, 1337, 1361,
            1366, 1368, 1370,
            1476, 1477, 1484, 1490, 1492, 1499, 1506, 1580, 1618, 1622, 1631, 1633, 1636, 1507, 1589,
            1607, 1639, 1642, 1775, 1777, 1778, 1794, 1816, 1817, 1821, 1827, 1835, 1849, 1850, 1853, 1856, 1872, 1875,
            1881, 1910, 1915, 1948]
test_ids = [str(x) for x in test_ids]


for t in tumor_types:
    paths = glob.glob(read_path + t + '/*.png')
    paths.sort()
    dst_dir = save_path + t + '/'
    count = 0

    for i in range(len(paths)):
        id =  os.path.basename(paths[i]).split('_')[0]

        if id in test_ids:
            count += 1
            copy2(paths[i], dst_dir)

    print('Tumor: ', t, ' #files=', count)