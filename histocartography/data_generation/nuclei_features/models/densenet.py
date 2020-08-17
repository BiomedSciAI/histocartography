import torch
import torch.nn as nn


class DenseNet(nn.Module):

    def __init__(self,
                 initial_filters=-1,
                 layers=-1,
                 depth=-1,
                 nb_dense_block=-1,
                 growth_rate=-1):
        super().__init__()

        self._initial_filters = initial_filters if initial_filters > 0 else 2 * growth_rate
        self._layers = layers
        self._depth = depth
        self._nb_dense_block = nb_dense_block
        self._growth_rate = growth_rate
        self._block_output_filters = []  # Useful during upsampling
        self.get_layers()

        self._init = self._build_init()
        self._center = self._build_center()
        self._exit = self._build_exit()

    def forward(self, x):
        x = self._init(x)
        x = self._center(x)
        x = self._exit(x)
        return x
    # enddef

    # layers in each dense block
    def get_layers(self):
        if not isinstance(self._layers, list):
            if self._layers == -1:
                assert (self._depth -
                        4) % 3 == 0, 'Depth must be 3 N + 4 if layers == -1'
                count = int((self._depth - 4) / 3)

                if self.bottleneck:
                    count = count // 2

                self._layers = [count for _ in range(self._nb_dense_block)]
            else:
                self._layers = [self._layers] * self._nb_dense_block
        # endif
    # enddef

    def _build_init(self):
        block = nn.Sequential(
            nn.Conv2d(
                3,
                self._initial_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(
                self._initial_filters),
            nn.ReLU())
        self._block_output_filters.append(self._initial_filters)
        return block
    # enddef

    def _build_center(self):
        num_features = self._initial_filters
        center = nn.ModuleList()

        for idx, num_layers in enumerate(self._layers):
            center.append(
                DenseBlock(
                    growth_rate=self._growth_rate,
                    num_layers=num_layers,
                    num_input_features=num_features))
            num_features = num_features + num_layers * self._growth_rate

            if idx != len(self._layers) - 1:
                center.append(
                    self._transition_layer(
                        num_input_features=num_features,
                        num_output_features=num_features // 2))
                num_features = num_features // 2

            self._block_output_filters.append(num_features)
        # endfor

        center.append(nn.BatchNorm2d(num_features))
        center.append(nn.ReLU())
        self._center_filters = num_features

        return nn.Sequential(*center)
    # enddef

    def _build_exit(self):
        exit = nn.ModuleList()

        #exit.append(nn.AdaptiveAvgPool2d((1, 1)))
        exit.append(Flatten())

        return nn.Sequential(*exit)
    # enddef

    def _transition_layer(self, num_input_features, num_output_features):
        layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(),
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
                padding=0),
        )
        return layer
    # enddef
# end


class DenseBlock(nn.Module):
    def __init__(self, growth_rate, num_layers, num_input_features):
        super(DenseBlock, self).__init__()
        self._growth_rate = growth_rate

        self._dense_block = nn.ModuleList()
        for i in range(num_layers):
            self._dense_block.append(self._DenseLayer(num_input_features))
            num_input_features += self._growth_rate
    # enddef

    def forward(self, x):
        for block in self._dense_block:
            x_convolved = block(x)
            x = torch.cat((x_convolved, x), 1)
        return x
    # enddef

    def _DenseLayer(self, num_input_features):
        block = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(),
            nn.Conv2d(
                num_input_features,
                self._growth_rate * 4,
                kernel_size=1,
                stride=1,
                bias=False),
            nn.BatchNorm2d(
                self._growth_rate * 4),
            nn.ReLU(),
            nn.Conv2d(
                self._growth_rate * 4,
                self._growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False))
        return block
    # enddef
# end


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
# end


if __name__ == '__main__':
    from torchsummary import summary
    model = DenseNet(initial_filters=-1,
                     layers=[1, 1, 1, 1, 1],
                     growth_rate=12)

    summary(model, (3, 72, 72))
