import glob
import random
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
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
    print('Data: train=', num_sample_train,
          ', val=', num_sample_val,
          ', test=', num_sample_test)

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
        self.base_data_split_path = config.base_data_split_path
        self.base_patches_path = config.base_patches_path
        self.evalmode = evalmode


    def get_patches_path(self):
        if len(self.magnifications) == 1:
            magnification = self.magnifications[0]

        patches_path = []
        for t in range(len(self.tumor_types)):
            troi_ids = self.get_troi_ids(tumor_type=self.tumor_types[t])

            for i in range(len(troi_ids)):
                paths = self.get_patches(tumor_type=self.tumor_types[t], troi_id=troi_ids[i], magnification=magnification)
                patches_path.append(paths)

        patches_path = sorted([item for sublist in patches_path for item in sublist])
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


    def get_patches(self, tumor_type, troi_id, magnification):
        paths = sorted(glob.glob(self.base_patches_path + tumor_type + '/' + magnification + '/' + troi_id + '_*.png'))
        return paths


class PatchCollate(object):
    def __init__(self, config):
        self.device = config.device

    def __call__(self, batch):
        data = [item[0] for item in batch]
        data = torch.stack(data).to(self.device)
        target = [item[1] for item in batch]
        target = torch.LongTensor(target).to(self.device)
        return data, target


class PatchDataLoader(data.Dataset):
    def __init__(self, evalmode, config, is_train):
        self.evalmode = evalmode
        self.is_train = is_train
        self.tumor_types = config.tumor_types
        self.tumor_labels = config.tumor_labels

        obj = GetPatchesPath(config=config, evalmode=self.evalmode)
        self.patches_path = obj.get_patches_path()

        unique = []
        for i in range(len(self.patches_path)):
            basename = os.path.basename(self.patches_path[i])
            if basename not in unique:
                unique.append(basename)

        self.transform = transforms.Compose([transforms.Resize(config.patch_scale),
                                             transforms.CenterCrop(config.patch_scale),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        patch_path = self.patches_path[index]
        img_ = Image.open(patch_path)
        img = self.data_transform(img_)
        img_.close()

        tumor_type = os.path.basename(patch_path).split('_')[1]
        label = self.tumor_labels[self.tumor_types.index(tumor_type)]
        return img, label

    def data_transform(self, img):
        if self.is_train:
            # Random horizontal flipping
            if random.random() > 0.5:
                img = TF.hflip(img)

            # Random vertical flipping
            if random.random() > 0.5:
                img = TF.vflip(img)

        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.patches_path)



