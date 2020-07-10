import glob
import os
import torch
import numpy as np
import dgl
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


# --------------------------
# Supporting functions
# --------------------------

# mode = [train, val, test]


def get_file_list(config, mode):
    patches_path = []
    for t in range(len(config.tumor_types)):
        troi_ids = get_troi_ids(
            config=config,
            mode=mode,
            tumor_type=config.tumor_types[t])

        for i in range(len(troi_ids)):
            paths = get_patches(
                config=config,
                tumor_type=config.tumor_types[t],
                troi_id=troi_ids[i])
            patches_path.append(paths)

    patches_path = sorted(
        [item for sublist in patches_path for item in sublist])
    return patches_path


def get_troi_ids(config, mode, tumor_type):
    trois = []
    filename = config.base_data_split_path + mode + '_list_' + tumor_type + '.txt'
    with open(filename, 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            if line != '':
                trois.append(line)

    return trois


def get_patches(config, tumor_type, troi_id):
    paths = sorted(
        glob.glob(
            config.base_patches_path +
            tumor_type +
            '/' +
            troi_id +
            '_*.png'))
    return paths


# -----------------------------------------------------------------------------------------------------------------------
# Patch data loader
# -----------------------------------------------------------------------------------------------------------------------

def get_transform(patch_size, patch_scale, is_pretrained, is_train=True):
    if is_train:
        if patch_size == patch_scale:
            if is_pretrained:
                data_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                data_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor()
                ])
        else:
            if is_pretrained:
                data_transform = transforms.Compose([
                    transforms.Resize(patch_scale),
                    transforms.CenterCrop(patch_scale),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                data_transform = transforms.Compose([
                    transforms.Resize(patch_scale),
                    transforms.CenterCrop(patch_scale),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor()
                ])
    else:
        if patch_size == patch_scale:
            if is_pretrained:
                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                data_transform = transforms.Compose([
                    transforms.ToTensor()
                ])
        else:
            if is_pretrained:
                data_transform = transforms.Compose([
                    transforms.Resize(patch_scale),
                    transforms.CenterCrop(patch_scale),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                data_transform = transforms.Compose([
                    transforms.Resize(patch_scale),
                    transforms.CenterCrop(patch_scale),
                    transforms.ToTensor()
                ])

    return data_transform


def patch_loaders(
        config,
        is_pretrained,
        random_seed=0,
        shuffle=True,
        pin_memory=False):
    batch_size = config.batch_size

    dataset_train = PatchDataLoader(
        mode='train',
        config=config,
        is_pretrained=is_pretrained,
        is_train=True)
    dataset_val = PatchDataLoader(
        mode='val',
        config=config,
        is_pretrained=is_pretrained,
        is_train=False)
    dataset_test = PatchDataLoader(
        mode='test',
        config=config,
        is_pretrained=is_pretrained,
        is_train=False)

    num_sample_train = len(dataset_train)
    indices_train = list(range(num_sample_train))
    num_sample_val = len(dataset_val)
    indices_val = list(range(num_sample_val))
    num_sample_test = len(dataset_test)
    indices_test = list(range(num_sample_test))
    print(
        'Data: train=',
        num_sample_train,
        ', val=',
        num_sample_val,
        ', test=',
        num_sample_test)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices_train)

    train_sampler = SubsetRandomSampler(indices_train)
    valid_sampler = SubsetRandomSampler(indices_val)
    test_sampler = SubsetRandomSampler(indices_test)

    train_loader = data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=pin_memory,
        collate_fn=patch_collate_fn)
    valid_loader = data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        sampler=valid_sampler,
        pin_memory=pin_memory,
        collate_fn=patch_collate_fn)
    test_loader = data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        sampler=test_sampler,
        pin_memory=pin_memory,
        collate_fn=patch_collate_fn)

    data_loaders = {
        'train': train_loader,
        'val': valid_loader,
        'test': test_loader
    }
    return data_loaders


def patch_collate_fn(batch):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    data = [item[0] for item in batch]
    data = torch.stack(data).to(device)
    target = [item[1] for item in batch]
    target = torch.LongTensor(target).to(device)
    return data, target


class PatchDataLoader(data.Dataset):
    def __init__(self, mode, config, is_pretrained, is_train):
        self.mode = mode
        self.classes = config.tumor_types
        self.class_to_idx = config.class_to_idx
        self.tumor_types = config.tumor_types
        self.base_patches_path = config.base_patches_path
        self.base_data_split_path = config.base_data_split_path

        self.file_list = get_file_list(config=config, mode=mode)
        self.data_transform = get_transform(
            config.patch_size, config.patch_scale, is_pretrained, is_train)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        img_ = Image.open(file_name)
        img = self.data_transform(img_)
        img_.close()

        class_label = os.path.basename(os.path.dirname(self.file_list[index]))
        label = self.class_to_idx[self.classes.index(class_label)]
        return img, label

    def __len__(self):
        return len(self.file_list)


# ----------------------
# TRoI data loader
# ----------------------


def troi_loaders(config, models, mode):
    if not (
        os.path.isfile(
            config.model_save_path +
            'train_data_' +
            mode +
            '.npz') and os.path.isfile(
            config.model_save_path +
            'val_data_' +
            mode +
            '.npz') and os.path.isfile(
                config.model_save_path +
                'test_data_' +
                mode +
            '.npz')):

        dataset_train = TROIDataLoader(config=config, models=models, mode='train')
        dataset_val = TROIDataLoader(config=config, models=models, mode='val')
        dataset_test = TROIDataLoader(config=config, models=models, mode='test')

        def get_data(loader):
            embedding = np.zeros(shape=(1, config.num_features))

            labels = []
            for i in range(len(loader)):
                emb, lb = loader.__getitem__(i)
                embedding = np.vstack(
                    (embedding, np.reshape(
                        emb, newshape=(
                            1, config.num_features))))
                labels.append(lb)

                if i % 100 == 0:
                    print(i)

            embedding = np.delete(embedding, 0, axis=0)
            return embedding, labels

        print('Extracting train features...')
        train_data, train_labels = get_data(dataset_train)
        print('Train: data=', train_data.shape, ' label=', len(train_labels))

        print('Extracting val features...')
        val_data, val_labels = get_data(dataset_val)
        print('Val: data=', val_data.shape, ' label=', len(val_labels))

        print('Extracting test features...')
        test_data, test_labels = get_data(dataset_test)
        print('Test: data=', test_data.shape, ' label=', len(test_labels))
        print('DONE')

        np.savez(
            config.model_save_path +
            'train_data_' +
            mode +
            '.npz',
            data=train_data,
            labels=train_labels)
        np.savez(
            config.model_save_path +
            'val_data_' +
            mode +
            '.npz',
            data=val_data,
            labels=val_labels)
        np.savez(
            config.model_save_path +
            'test_data_' +
            mode +
            '.npz',
            data=test_data,
            labels=test_labels)

        return train_data, train_labels, val_data, val_labels, test_data, test_labels
    else:
        data = np.load(config.model_save_path + 'train_data_' + mode + '.npz')
        train_data = data['data']
        train_labels = data['labels']
        print('Train: data=', train_data.shape, ' label=', len(train_labels))

        data = np.load(config.model_save_path + 'val_data_' + mode + '.npz')
        val_data = data['data']
        val_labels = data['labels']
        print('Val: data=', val_data.shape, ' label=', len(val_labels))

        data = np.load(config.model_save_path + 'test_data_' + mode + '.npz')
        test_data = data['data']
        test_labels = data['labels']
        print('Test: data=', test_data.shape, ' label=', len(test_labels))

        return train_data, train_labels, val_data, val_labels, test_data, test_labels


class TROIDataLoader(data.Dataset):
    def __init__(self, config, models, mode):
        self.base_img_path = config.base_img_path
        self.base_patches_path = config.base_patches_path
        self.base_data_split_path = config.base_data_split_path
        self.tumor_types = config.tumor_types
        self.classes = config.tumor_types
        self.class_to_idx = config.class_to_idx
        self.weight_merge = config.weight_merge
        self.num_classes = config.num_classes
        self.num_features = config.num_features
        self.device = config.device
        self.as_graph = config.as_graph
        self.mode = mode

        self.troi_list = self.get_troi_list(config=config)
        self.data_transform = get_transform(
            config.patch_size,
            config.patch_scale,
            config.is_pretrained,
            is_train=False)

        self.embedding_model, self.classifier_model = models
        self.embedding_model = self.embedding_model.to(self.device)
        self.classifier_model = self.classifier_model.to(self.device)
        self.embedding_model.eval()
        self.classifier_model.eval()

        self.avgpool = torch.nn.AdaptiveAvgPool1d(self.num_features)

    def get_troi_list(self, config):
        troi_ids = []
        for t in range(len(self.tumor_types)):
            ids = get_troi_ids(
                config=config,
                mode=self.mode,
                tumor_type=self.tumor_types[t])
            troi_ids.append(ids)

        troi_ids = [item for sublist in troi_ids for item in sublist]
        return troi_ids

    def __getitem__(self, index):
        class_label = self.troi_list[index].split('_')[1]
        basename = self.troi_list[index]
        label = self.class_to_idx[self.classes.index(class_label)]
        patches_list = glob.glob(
            self.base_patches_path +
            class_label +
            '/' +
            basename +
            '_*.png')

        patches = []
        for i in range(len(patches_list)):
            img_ = Image.open(patches_list[i])
            img = self.data_transform(img_)
            img_.close()
            patches.append(img)

        if len(patches) != 0:
            patches = torch.stack(patches)
            patches = patches.to(self.device)

            with torch.no_grad():
                embedding = self.embedding_model(patches)
                embedding = embedding.squeeze(dim=2)
                embedding = embedding.squeeze(dim=2)
                self.dim = embedding.shape[1]
                del patches

                if self.weight_merge:
                    probabilities = self.classifier_model(embedding)
                else:
                    probabilities = torch.ones(
                        embedding.shape[0], self.num_classes)

                emb = embedding.repeat(1, self.num_classes).to(self.device)
                prob = probabilities.repeat_interleave(
                    self.dim, dim=1).to(self.device)

                embedding_wt = (emb * prob).flatten()
                embedding_wt = embedding_wt.unsqueeze(dim=0)
                embedding_wt = embedding_wt.unsqueeze(dim=0)
                embedding_wt = self.avgpool(embedding_wt).squeeze().cpu().detach().numpy()
                embedding_wt = embedding_wt.view(size=(-1, self.num_features))

                if self.as_graph:
                    graph = dgl.DGLGraph()
                    graph.add_nodes(embedding.shape[0])
                    graph.ndata['h'] = embedding_wt
                    centroid = []
                    # build topology based on the centroid or the patch bounding box
                    graph.add_edges([0, 1], [0, 1])
                    return graph, label

                return embedding_wt, label

        else:
            print('ERROR: Troi with empty patches', self.troi_list[index])
            exit()

    def __len__(self):
        return len(self.troi_list)
