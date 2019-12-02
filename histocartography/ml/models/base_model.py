from torch.nn import Module


class BaseModel(Module):

    def __init__(self):
        """
        Base model constructor.
        """
        super(BaseModel, self).__init__()

    def forward(self, graphs):
        """
        Forward pass
        :param graphs:
        """
        raise NotImplementedError('Implementation in subclasses.')
