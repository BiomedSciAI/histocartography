from torch.nn import Module


class BaseLayer(Module):

    def __init__(self, node_dim, hidden_dim, out_dim, act, layer_id):
        """
        Base layer constructor.
        """
        super(BaseLayer, self).__init__()
        self.layer_id = layer_id

    def forward(self, g, h):
        """
        Forward pass
        :param g:
        :param h:
        """
        raise NotImplementedError('Implementation in subclasses.')
