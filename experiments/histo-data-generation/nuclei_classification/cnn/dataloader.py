import glob
import random
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from utils import *
import copy


def patch_loaders(config, args, pin_memory=False, is_sampler=False,  is_oversample=False):
    batch_size = args.batch_size
    num_workers = args.workers
    patch_collate_fn = PatchCollate(config)

    dataset_train = PatchDataLoader(
        evalmode='train',
        is_oversample=is_oversample,
        config=config)
    dataset_val = PatchDataLoader(
        evalmode='val',
        is_oversample=is_oversample,
        config=config)
    dataset_test = PatchDataLoader(
        evalmode='test',
        is_oversample=is_oversample,
        config=config)

    num_sample_train = len(dataset_train)
    num_sample_val = len(dataset_val)
    num_sample_test = len(dataset_test)
    print('Data: train=', num_sample_train,
          ', val=', num_sample_val,
          ', test=', num_sample_test)

    # Get class weights:
    patches_count = dataset_train.patches_count
    weight = 1. / patches_count
    weight = weight / np.sum(weight)

    print('Patch count: ', patches_count)
    print('Class weights: ', weight)

    # Get nuclei class weights
    if is_sampler:
        patches_label = dataset_train.patches_label
        samples_weight = torch.from_numpy(weight[patches_label.astype(int)])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader = data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            pin_memory=pin_memory,
            collate_fn=patch_collate_fn,
            shuffle=True,
            num_workers=num_workers,
            sampler=sampler)

    else:
        train_loader = data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            pin_memory=pin_memory,
            collate_fn=patch_collate_fn,
            shuffle=True,
            num_workers=num_workers)

    valid_loader = data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        pin_memory=pin_memory,
        collate_fn=patch_collate_fn,
        shuffle=False,
        num_workers=num_workers)
    test_loader = data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        pin_memory=pin_memory,
        collate_fn=patch_collate_fn,
        shuffle=False,
        num_workers=num_workers)

    data_loaders = {
        'train': train_loader,
        'val': valid_loader,
        'test': test_loader,
    }
    return data_loaders, torch.from_numpy(weight).float()


class GetPatchesPath:
    def __init__(self, config, evalmode, nuclei_types, nuclei_labels, is_oversample):
        self.config = config
        self.evalmode = evalmode
        self.nuclei_types = nuclei_types
        self.nuclei_labels = nuclei_labels
        self.is_oversample = is_oversample

    def get_samplenames(self):
        self.samplenames = []
        with open(self.config.base_data_split_path + self.evalmode + '.txt', 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                if line != '':
                    self.samplenames.append(line)

    def get_patches_info_per_type(self, nuclei_type):
        patches_path = []
        for s in self.samplenames:
            paths = glob.glob(self.config.base_patches_path + nuclei_type + '/' + s + '_*.png')
            patches_path += paths
        return patches_path

    def get_oversampled_samples(self, paths):
        counts = np.asarray([len(x) for x in paths])
        max_count = np.max(counts)

        for t in range(len(self.nuclei_types)):
            diff = max_count - counts[t]

            np.random.seed(t)
            idx = np.random.choice(np.arange(paths[t]), diff.astype(int), replace=True)
            diff = [paths[t][x] for x in idx]
            paths[t] += diff
        return paths

    def get_patches_path(self):
        self.get_samplenames()

        paths_ = [[] for i in range(len(self.nuclei_types))]
        # Get per type patches path and count
        for t in range(len(self.nuclei_types)):
            paths_[t] = self.get_patches_info_per_type(self.nuclei_types[t])

        if self.evalmode == 'train' and self.is_oversample:
            paths_ = self.get_oversampled_samples(paths_)

        labels_ = [[] for i in range(len(self.nuclei_types))]
        count_ = []
        for t in range(len(paths_)):
            labels_[t] = (np.ones(len(paths_[t])) * self.nuclei_labels[t]).tolist()
            count_.append(len(paths_[t]))

        paths = [item for sublist in paths_ for item in sublist]
        labels = [item for sublist in labels_ for item in sublist]

        labels = np.asarray(labels)
        count = np.asarray(count_)

        return paths, labels, count


class PatchCollate(object):
    def __init__(self, config):
        self.device = config.device

    def __call__(self, batch):
        data = [item[0] for item in batch]
        data = torch.stack(data)
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)
        return data, target


class PatchDataLoader(data.Dataset):
    def __init__(self, evalmode, is_oversample, config):
        self.evalmode = evalmode
        self.nuclei_types = copy.deepcopy(config.nuclei_types)
        self.nuclei_labels = copy.deepcopy(config.nuclei_labels)
        self.nuclei_types = [x.lower() for x in self.nuclei_types]

        # Remove nuclei type = 'NA'
        idx = self.nuclei_labels.index(-1)
        del self.nuclei_labels[-idx]
        del self.nuclei_types[-idx]

        obj = GetPatchesPath(config=config,
                             evalmode=self.evalmode,
                             nuclei_types=self.nuclei_types,
                             nuclei_labels=self.nuclei_labels,
                             is_oversample=is_oversample)

        self.patches_path, self.patches_label, self.patches_count = obj.get_patches_path()
        # labels: 0=normal, 1=atypical, 2=tumor, 3=stromal, 4=lymphocyte, 5=dead

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        patch_path = self.patches_path[index]
        img_ = Image.open(patch_path)
        img = self.data_transform(img_)
        img_.close()

        nuclei_type = os.path.basename(patch_path).split('_')[3]
        label = self.nuclei_labels[self.nuclei_types.index(nuclei_type)]
        return img, label

    def data_transform(self, img):
        if self.evalmode == 'train':
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
