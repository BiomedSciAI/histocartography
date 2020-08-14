import glob
import numpy as np
import torch.utils.data as data
from PIL import Image
from utils import *
import random
from torchvision import transforms
import torchvision.transforms.functional as TF


class Patch_DataLoader(data.Dataset):
    def __init__(self, config, mode):
        self.nuclei_patches_path = glob.glob(config.nuclei_patch_path + mode + '/*.png')
        self.nuclei_mask_path = config.nuclei_mask_path + mode + '/'
        self.data_transform = transforms.ToTensor()

    def __getitem__(self, index):
        patch_file_path = self.nuclei_patches_path[index]
        mask_file_path = self.nuclei_mask_path + os.path.basename(patch_file_path)
        filename = os.path.basename(patch_file_path).split('.')[0]

        patch = Image.open(patch_file_path)
        mask = Image.open(mask_file_path)

        # Random horizontal flipping
        if random.random() > 0.5:
            patch = TF.hflip(patch)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            patch = TF.vflip(patch)
            mask = TF.vflip(mask)

        patch = self.data_transform(patch)
        mask = self.data_transform(mask)

        return (patch, mask, filename)

    def __len__(self):
        return len(self.nuclei_patches_path)
#end


def patch_loaders(config, random_seed=0, shuffle=True, pin_memory=True):
    batch_size = config.batch_size

    dataset_train = Patch_DataLoader(config=config, mode='train')
    dataset_val = Patch_DataLoader(config=config, mode='val')
    dataset_test = Patch_DataLoader(config=config, mode='test')

    num_sample_train = len(dataset_train)
    indices_train = list(range(num_sample_train))
    num_sample_val = len(dataset_val)
    indices_val = list(range(num_sample_val))
    num_sample_test = len(dataset_test)
    indices_test = list(range(num_sample_test))
    print('Data: train=', num_sample_train, ', val=', num_sample_val, ', test=', num_sample_test)

    '''
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices_train)

    train_sampler = SubsetRandomSampler(indices_train)
    valid_sampler = SubsetRandomSampler(indices_val)
    test_sampler = SubsetRandomSampler(indices_test)

    train_loader = data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, pin_memory=pin_memory)
    valid_loader = data.DataLoader(dataset_val, batch_size=batch_size, sampler=valid_sampler, pin_memory=pin_memory)
    test_loader = data.DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler, pin_memory=pin_memory)
    #'''

    train_loader = data.DataLoader(dataset_train, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    valid_loader = data.DataLoader(dataset_val, batch_size=batch_size, pin_memory=pin_memory, shuffle=False)
    test_loader = data.DataLoader(dataset_test, batch_size=batch_size, pin_memory=pin_memory, shuffle=False)


    data_loaders = {
        'train': train_loader,
        'val': valid_loader,
        'test': test_loader
    }
    return data_loaders
#enddef


def view_patch_dataloader(data_iter):
    for i in range(len(data_iter)):
        p, m = data_iter.__getitem__(i)
        p = p.cpu().detach().numpy()
        p = np.moveaxis(p, 0, -1)
        plot(p)

        m = m.cpu().detach().numpy().astype(np.uint8)
        m = np.squeeze(m)
        plot(m)

        if i == 2:
            exit()
    #endfor
#enddef

