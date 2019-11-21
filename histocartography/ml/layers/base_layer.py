from torch.nn import Module


class BaseLayer(Module):

    def __init__(self, node_dim, hidden_dim, out_dim, act, layer_id):
        """
        Base layer constructor.
        """

    def forward(self, g, h):
        """
        Forward pass
        :param g:
        :param h:
        """
        raise NotImplementedError('Implementation in subclasses.')
