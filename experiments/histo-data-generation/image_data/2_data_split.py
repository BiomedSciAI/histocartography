import argparse
import os
import glob


def select_ids(split_number):
    if split_number == 1:
        val_ids = [
            283,
            743,
            747,
            750,
            754,
            760,
            1231,
            1232,
            1242,
            1253,
            1289,
            1292,
            1293,
            1366,
            1358]
        test_ids = [
            281,
            286,
            291,
            301,
            739,
            753,
            757,
            1286,
            1255,
            1257,
            1296,
            1319,
            1337,
            1368,
            1369]
    elif split_number == 2:
        val_ids = [
            264,
            288,
            292,
            295,
            738,
            740,
            755,
            756,
            762,
            1235,
            1238,
            1283,
            1356,
            1360]
        test_ids = [
            281,
            286,
            291,
            301,
            739,
            753,
            757,
            1286,
            1255,
            1257,
            1296,
            1319,
            1337,
            1368,
            1369]
    elif split_number == 3:
        val_ids = [
            265,
            297,
            305,
            311,
            735,
            736,
            737,
            745,
            752,
            759,
            761,
            1272,
            1288,
            1263,
            1271,
            1325,
            1330,
            1334,
            1335,
            1395,
            1416]
        test_ids = [
            281,
            286,
            291,
            301,
            739,
            753,
            757,
            1286,
            1255,
            1257,
            1296,
            1319,
            1337,
            1368,
            1369]
    elif split_number == 4:
        val_ids = [
            281,
            286,
            291,
            301,
            739,
            753,
            757,
            1286,
            1255,
            1257,
            1296,
            1319,
            1337,
            1368,
            1369]
        test_ids = [
            283,
            743,
            747,
            750,
            754,
            760,
            1231,
            1232,
            1242,
            1253,
            1289,
            1292,
            1293,
            1366,
            1358]
    elif split_number == 5:
        val_ids = [
            264,
            288,
            292,
            295,
            738,
            740,
            755,
            756,
            762,
            1235,
            1238,
            1283,
            1356,
            1360]
        test_ids = [
            283,
            743,
            747,
            750,
            754,
            760,
            1231,
            1232,
            1242,
            1253,
            1289,
            1292,
            1293,
            1366,
            1358]
    elif split_number == 6:
        val_ids = [
            265,
            297,
            305,
            311,
            735,
            736,
            737,
            745,
            752,
            759,
            761,
            1272,
            1288,
            1263,
            1271,
            1325,
            1330,
            1334,
            1335,
            1395,
            1416]
        test_ids = [
            283,
            743,
            747,
            750,
            754,
            760,
            1231,
            1232,
            1242,
            1253,
            1289,
            1292,
            1293,
            1366,
            1358]
    elif split_number == 7:
        val_ids = [
            281,
            286,
            291,
            301,
            739,
            753,
            757,
            1286,
            1255,
            1257,
            1296,
            1319,
            1337,
            1368,
            1369]
        test_ids = [
            264,
            288,
            292,
            295,
            738,
            740,
            755,
            756,
            762,
            1235,
            1238,
            1283,
            1356,
            1360]
    elif split_number == 8:
        val_ids = [
            283,
            743,
            747,
            750,
            754,
            760,
            1231,
            1232,
            1242,
            1253,
            1289,
            1292,
            1293,
            1366,
            1358]
        test_ids = [
            264,
            288,
            292,
            295,
            738,
            740,
            755,
            756,
            762,
            1235,
            1238,
            1283,
            1356,
            1360]
    elif split_number == 9:
        val_ids = [
            265,
            297,
            305,
            311,
            735,
            736,
            737,
            745,
            752,
            759,
            761,
            1272,
            1288,
            1263,
            1271,
            1325,
            1330,
            1334,
            1335,
            1395,
            1416]
        test_ids = [
            264,
            288,
            292,
            295,
            738,
            740,
            755,
            756,
            762,
            1235,
            1238,
            1283,
            1356,
            1360]
    elif split_number == 10:
        val_ids = [
            281,
            286,
            291,
            301,
            739,
            753,
            757,
            1286,
            1255,
            1257,
            1296,
            1319,
            1337,
            1368,
            1369]
        test_ids = [
            265,
            297,
            305,
            311,
            735,
            736,
            737,
            745,
            752,
            759,
            761,
            1272,
            1288,
            1263,
            1271,
            1325,
            1330,
            1334,
            1335,
            1395,
            1416]
    elif split_number == 11:
        val_ids = [
            283,
            743,
            747,
            750,
            754,
            760,
            1231,
            1232,
            1242,
            1253,
            1289,
            1292,
            1293,
            1366,
            1358]
        test_ids = [
            265,
            297,
            305,
            311,
            735,
            736,
            737,
            745,
            752,
            759,
            761,
            1272,
            1288,
            1263,
            1271,
            1325,
            1330,
            1334,
            1335,
            1395,
            1416]
    elif split_number == 12:
        val_ids = [
            264,
            288,
            292,
            295,
            738,
            740,
            755,
            756,
            762,
            1235,
            1238,
            1283,
            1356,
            1360]
        test_ids = [
            265,
            297,
            305,
            311,
            735,
            736,
            737,
            745,
            752,
            759,
            761,
            1272,
            1288,
            1263,
            1271,
            1325,
            1330,
            1334,
            1335,
            1395,
            1416]

    val_ids = [str(x) for x in val_ids]
    test_ids = [str(x) for x in test_ids]
    return val_ids, test_ids
# enddef


def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def save_to_txt(list, savepath):
    with open(savepath, 'w') as output:
        for row in list:
            basename = os.path.basename(row).split('.')[0]
            output.write(basename + '\n')
# enddef


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


# -----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--split_number', type=int, default=1, help='Split number')
args = parser.parse_args()
val_ids, test_ids = select_ids(split_number=args.split_number)

val_ids = [
    283,
    743,
    747,
    750,
    754,
    760,
    1231,
    1232,
    1242,
    1253,
    1289,
    1292,
    1293,
    1366,
    1358]
val_ids = [str(x) for x in val_ids]

tumor_types = [
    'benign',
    'pathologicalbenign',
    'udh',
    'adh',
    'fea',
    'dcis',
    'malignant']

base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/'
save_path = base_path + 'data_splits/data_split_' + \
    str(args.split_number) + '/'
create_directory(save_path)

train_total = 0
val_total = 0
test_total = 0

for t in range(len(tumor_types)):
    files = sorted(
        glob.glob(
            base_path +
            'Images_norm/' +
            tumor_types[t] +
            '/*.png'))
    train = []
    valid = []
    test = []

    for i in range(len(files)):
        basename = os.path.basename(files[i]).split('.')[0]
        id = basename.split('_')[0]

        if id in test_ids:
            test.append(files[i])
        elif id in val_ids:
            valid.append(files[i])
        else:
            train.append(files[i])
    # endfor

    check_unique(train)
    check_unique(valid)
    check_unique(test)

    save_to_txt(train, save_path + 'train_list_' + tumor_types[t] + '.txt')
    save_to_txt(valid, save_path + 'val_list_' + tumor_types[t] + '.txt')
    save_to_txt(test, save_path + 'test_list_' + tumor_types[t] + '.txt')
    print(
        tumor_types[t],
        ': train=',
        len(train),
        'val=',
        len(valid),
        'test=',
        len(test))

    train_total += len(train)
    val_total += len(valid)
    test_total += len(test)
# endfor


print('TOTAL: train=', train_total, 'val=', val_total, 'test=', test_total)
