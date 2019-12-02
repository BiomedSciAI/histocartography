from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, cuda=False):
        """
        Base dataset constructor.
        """
        self.cuda = cuda

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        raise NotImplementedError('Implementation in subclasses.')

    def __len__(self):
        """

        :return:
        """
        raise NotImplementedError('Implementation in subclasses.')
