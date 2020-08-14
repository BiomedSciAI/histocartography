import glob
import random
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from utils import *


def patch_loaders(config,
        pin_memory=False):
    batch_size = config.batch_size

    dataset_train = PatchDataLoader(
        evalmode='train',
        config=config,
        is_train=True)
    dataset_val = PatchDataLoader(
        evalmode='val',
        config=config,
        is_train=False)
    dataset_test = PatchDataLoader(
        evalmode='test',
        config=config,
        is_train=False)

    num_sample_train = len(dataset_train)
    num_sample_val = len(dataset_val)
    num_sample_test = len(dataset_test)
    print('Data: train=', num_sample_train, ', val=', num_sample_val, ', test=', num_sample_test)

    patch_collate_fn = PatchCollate(config)

    train_loader = data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        pin_memory=pin_memory,
        collate_fn=patch_collate_fn,
        shuffle=True)
    valid_loader = data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        pin_memory=pin_memory,
        collate_fn=patch_collate_fn,
        shuffle=False)
    test_loader = data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        pin_memory=pin_memory,
        collate_fn=patch_collate_fn,
        shuffle=False)

    data_loaders = {
        'train': train_loader,
        'val': valid_loader,
        'test': test_loader
    }
    return data_loaders


class GetPatchesPath:
    def __init__(self, config, evalmode):
        self.tumor_types = config.tumor_types
        self.magnifications = config.magnifications
        self.tumor_types = config.tumor_types
        self.base_data_split_path = config.base_data_split_path
        self.base_patches_path = config.base_patches_path
        self.evalmode = evalmode


    def get_patches_path(self):
        patches_path = [[] for i in range(len(self.magnifications))]

        for t in range(len(self.tumor_types)):
            troi_ids = self.get_troi_ids(tumor_type=self.tumor_types[t])

            for i in range(len(troi_ids)):
                for m in range(len(self.magnifications)):
                    paths = self.read_patches_path(tumor_type=self.tumor_types[t], troi_id=troi_ids[i], magnification=self.magnifications[m])
                    patches_path[m].append(paths)

        for m in range(len(self.magnifications)):
            patches_path[m] = sorted([item for sublist in patches_path[m] for item in sublist])

        return patches_path


    def get_troi_ids(self, tumor_type):
        troi_ids = []
        filename = self.base_data_split_path + self.evalmode + '_list_' + tumor_type + '.txt'
        with open(filename, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                if line != '':
                    troi_ids.append(line)

        return troi_ids


    def read_patches_path(self, tumor_type, troi_id, magnification):
        paths = sorted(glob.glob(self.base_patches_path + tumor_type + '/' + magnification + '/' + troi_id + '_*.png'))
        return paths


class PatchCollate(object):
    def __init__(self, config):
        self.magnifications = config.magnifications
        self.device = config.device

    def __call__(self, batch):
        data = [[] for i in range(len(self.magnifications))]
        target = []

        for i in range(len(batch)):
            for m in range(len(self.magnifications)):
                data[m].append(batch[i][0][m])
            target.append(batch[i][1])

        for m in range(len(self.magnifications)):
            data[m] = torch.stack(data[m]).to(self.device)

        target = torch.LongTensor(target).to(self.device)
        return data, target


class PatchDataLoader(data.Dataset):
    def __init__(self, evalmode, config, is_train):
        self.evalmode = evalmode
        self.is_train = is_train
        self.tumor_types = config.tumor_types
        self.tumor_labels = config.tumor_labels
        self.magnifications = config.magnifications

        obj = GetPatchesPath(config=config, evalmode=evalmode)
        self.patches_path = obj.get_patches_path()

        self.transform = transforms.Compose([transforms.Resize(config.patch_scale),
                                             transforms.CenterCrop(config.patch_scale),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        img = []
        for m in range(len(self.magnifications)):
            img.append(Image.open(self.patches_path[m][index]))

        img = self.data_transform(img)
        tumor_type = os.path.basename(self.patches_path[0][index]).split('_')[1]
        label = self.tumor_labels[self.tumor_types.index(tumor_type)]
        return img, label

    def data_transform(self, img):
        if self.is_train:
            # Random horizontal flipping
            if random.random() > 0.5:
                for m in range(len(self.magnifications)):
                    img[m] = TF.hflip(img[m])

            # Random vertical flipping
            if random.random() > 0.5:
                for m in range(len(self.magnifications)):
                    img[m] = TF.vflip(img[m])

        for m in range(len(self.magnifications)):
            img[m] = self.transform(img[m])
        return img

    def __len__(self):
        return len(self.patches_path[0])



