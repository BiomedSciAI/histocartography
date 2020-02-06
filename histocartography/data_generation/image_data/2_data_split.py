import os
import glob
import numpy as np

def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def save_to_txt(list, savepath):
    with open(savepath, 'w') as output:
        for row in list:
            basename = os.path.basename(row).split('.')[0]
            output.write(basename + '\n')
#enddef

def check_unique(list):
    check = []
    for i in range(len(list)):
        if list[i] not in check:
            check.append(list[i])
    #endfor
    if len(list) == len(check):
        print('All unique')
    else:
        print('ERROR')
        exit()


#-----------------------------------------------------------------------------------------------------------------------
test_ids = [281, 286, 291, 301, 739, 753, 757, 1286, 1255, 1257, 1296, 1319, 1337, 1368, 1369]
test_ids = [str(x) for x in test_ids]

tumor_types = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']

base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/'
save_path = base_path + 'data_split/'
create_directory(save_path)

for t in range(len(tumor_types)):
    files = glob.glob(base_path + 'Images_norm/' + tumor_types[t] + '/*.png')
    files.sort()
    train_n_val = []
    test = []

    for i in range(len(files)):
        basename = os.path.basename(files[i]).split('.')[0]
        id = basename.split('_')[0]

        if id not in test_ids:
            train_n_val.append(files[i])
        else:
            test.append(files[i])
    #endfor

    np.random.seed(t)
    idx = np.random.choice(len(train_n_val), int(0.8 * len(train_n_val)), replace=False)
    train = [train_n_val[x] for x in idx]

    idx = np.delete(np.arange(len(train_n_val)), idx)
    valid = [train_n_val[x] for x in idx]

    check_unique(train)
    check_unique(valid)
    check_unique(test)

    save_to_txt(train, save_path + 'train_list_' + tumor_types[t] + '.txt')
    save_to_txt(valid, save_path + 'val_list_' + tumor_types[t] + '.txt')
    save_to_txt(test, save_path + 'test_list_' + tumor_types[t] + '.txt')
    print(tumor_types[t], ': train=', len(train),  'val=', len(valid), 'test=', len(test))
