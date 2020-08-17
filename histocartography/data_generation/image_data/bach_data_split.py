import glob
import os
import numpy as np


def save_to_txt(list, savepath):
    with open(savepath, 'w') as output:
        for row in list:
            output.write(row + '\n')


def check_unique(list):
    check = []
    for i in range(len(list)):
        if list[i] not in check:
            check.append(list[i])
    # endfor
    if len(list) == len(check):
        print('All unique')
    else:
        print('ERROR')
        exit()


base_path = '/dataT/pus/histocartography/Data/BACH/train/'
save_path = base_path + 'data_split_cv/data_split_1/'

tumor_types = ['benign', 'pathologicalbenign', 'dcis', 'malignant']

for tumor_type in tumor_types:
    filelist = glob.glob(base_path + 'Images_norm/' + tumor_type + '/*.png')
    filelist = [os.path.basename(x).split('.')[0] for x in filelist]

    train_idx = np.random.choice(
        np.arange(
            len(filelist)), int(
            0.8 * len(filelist)), replace=False)
    val_idx = np.delete(np.arange(len(filelist)), train_idx)

    train_files = sorted([filelist[x] for x in train_idx])

    val_files = [filelist[x] for x in val_idx]
    val_files.sort()

    print('CHECK UNIQUE')
    check_unique(train_files)
    check_unique(val_files)

    save_to_txt(train_files, save_path + 'train_list_' + tumor_type + '.txt')
    save_to_txt(val_files, save_path + 'val_list_' + tumor_type + '.txt')
    save_to_txt(val_files, save_path + 'test_list_' + tumor_type + '.txt')
