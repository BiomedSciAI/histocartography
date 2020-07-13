import glob
import os
import torch
import numpy as np
import dgl
import pickle
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn.pytorch.factory import KNNGraph
from tqdm import tqdm 
from histocartography.dataloader.constants import get_tumor_type_to_label, ALL_DATASET_NAMES, get_dataset_white_list

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
        self.tumor_type_to_label = config.tumor_type_to_label
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
        label = self.tumor_type_to_label[class_label]
        return img, label

    def __len__(self):
        return len(self.file_list)


# ----------------------
# TRoI data loader
# ----------------------


def troi_loaders(config, models, mode):

    def collate(batch):
        data = dgl.batch([example[0] for example in batch])
        labels = torch.LongTensor([example[1] for example in batch]).to(config.device)
        return data, labels

    dataset_train = TROIDataLoader(config=config, models=models, mode='train')
    dataset_val = TROIDataLoader(config=config, models=models, mode='val')
    dataset_test = TROIDataLoader(config=config, models=models, mode='test')

    # create dataloader from the dataset 
    train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate
        )
    val_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate
        )
    test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate
        )

    return train_dataloader, val_dataloader, test_dataloader


class TROIDataLoader(data.Dataset):
    def __init__(self, config, models, mode):
        self.base_img_path = config.base_img_path
        self.base_patches_path = config.base_patches_path
        self.base_data_split_path = config.base_data_split_path
        self.tumor_type_to_label = get_tumor_type_to_label(config.class_split)
        self.weight_merge = config.weight_merge
        self.num_classes = config.num_classes
        self.num_features = config.num_features
        self.device = config.device
        self.as_graph = True
        self.mode = mode
        self.split = config.split 
        self.in_ram = config.in_ram 
        self.patch_size = config.patch_size

        self.troi_list = self.get_troi_list(config=config)
        self.data_transform = get_transform(
            config.patch_size,
            config.patch_scale,
            config.is_pretrained,
            is_train=False)

        self.embedding_model, self.classifier_model = models
        self.embedding_model = self.embedding_model.to(self.device)
        self.embedding_model.eval()
        if self.classifier_model is not None:
            self.classifier_model = self.classifier_model.to(self.device)
            self.classifier_model.eval()

        if self.in_ram and self.as_graph:
            self.all_graphs = []
            self.all_labels = []
            print('In RAM data loading & processing...')
            for index in tqdm(range(len(self.troi_list))):
                class_label = self.troi_list[index].split('_')[1]
                label = self.tumor_type_to_label[class_label]
                basename = self.troi_list[index]
                # label = self.class_to_idx[self.classes.index(class_label)]
                patches_list = glob.glob(
                    os.path.join(
                        self.base_patches_path,
                        class_label,
                        config.patch_size,
                        basename + '_*.png'
                    )
                )

                patches = []
                patch_coords = []
                for i in range(len(patches_list)):
                    img_ = Image.open(patches_list[i])
                    img = self.data_transform(img_)
                    img_.close()
                    patches.append(img)
                    x = patches_list[i].split('_')[-2]
                    y = patches_list[i].split('_')[-1].split('.')[0]
                    patch_coords.append([float(x), float(y)])

                num_patches = len(patches)
                if num_patches != 0:
                    patches = torch.stack(patches)
                    patches = patches.to(self.device)

                    with torch.no_grad():
                        embedding = self.embedding_model(patches)
                        embedding = embedding.squeeze(dim=2)
                        embedding = embedding.squeeze(dim=2)
                        self.dim = embedding.shape[1]
                        del patches
                        self.graph_builder = KNNGraph(k=min(8, num_patches))
                        graph = self.graph_builder(torch.tensor(patch_coords))
                        graph.ndata['h'] = embedding
                        self.all_graphs.append(graph)
                        self.all_labels.append(label)

        if not self.as_graph:
            self.avgpool = torch.nn.AdaptiveAvgPool1d(self.num_features)

    def get_troi_list(self, config):
        troi_ids = []
        datasets = get_dataset_white_list(config.class_split)
        for t in range(len(datasets)):
            ids = get_troi_ids(
                config=config,
                mode=self.mode,
                tumor_type=datasets[t])
            troi_ids.append(ids)

        troi_ids = [item for sublist in troi_ids for item in sublist]
        return troi_ids

    def __getitem__(self, index):
        if self.in_ram:
            return self.all_graphs[index], self.all_labels[index]

        class_label = self.troi_list[index].split('_')[1]
        label = self.tumor_type_to_label[class_label]
        basename = self.troi_list[index]
        patches_list = glob.glob(
            os.path.join(
                self.base_patches_path,
                class_label,
                self.patch_size,
                basename + '_*.png'
            )
        )

        patches = []
        patch_coords = []
        for i in range(len(patches_list)):
            img_ = Image.open(patches_list[i])
            img = self.data_transform(img_)
            img_.close()
            patches.append(img)
            x = patches_list[i].split('_')[-2]
            y = patches_list[i].split('_')[-1].split('.')[0]
            patch_coords.append([float(x), float(y)])

        num_patches = len(patches)
        if num_patches != 0:
            patches = torch.stack(patches)
            patches = patches.to(self.device)

            with torch.no_grad():
                embedding = self.embedding_model(patches)
                embedding = embedding.squeeze(dim=2)
                embedding = embedding.squeeze(dim=2)
                self.dim = embedding.shape[1]
                del patches

                if self.as_graph:
                    self.graph_builder = KNNGraph(k=min(8, num_patches))
                    graph = self.graph_builder(torch.tensor(patch_coords))
                    graph.ndata['h'] = embedding
                    return graph, label

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
                embedding_wt = self.avgpool(embedding_wt)

                return embedding_wt, label

        else:
            print('ERROR: Troi with empty patches', self.troi_list[index])
            exit()

    def __len__(self):
        return len(self.troi_list)
