import torch
import torch.nn as nn
import torch.nn.functional as F 
import dgl


class HoverNet(nn.Module):

    def __init__(self, nr_types=6):
        """

        """

        super(HoverNet, self).__init__()

        self.nr_types = nr_types

        # define encoder 
        self.encode = Encoder()

        # define decoder(s)
        self.decode_np = Decoder()
        self.decode_hv = Decoder()
        self.decode_tp = Decoder()

        # nuclei type pixel (TP)
        self.conv_out_tp = Conv2dWithActivation(64, self.nr_types, 1, activation=None, padding=1, bias=True)
        self.preact_out_np = BNReLU(64)

        # nuclei pixels (NP)
        self.conv_out_np = Conv2dWithActivation(64, 2, 1, activation=None, padding=1, bias=True)
        self.preact_out_hv = BNReLU(64)

        # horizontal-vertival (HV)
        self.conv_out_hv = Conv2dWithActivation(64, 2, 1, activation=None, padding=1, bias=True)
        self.preact_out_tp = BNReLU(64)
 
    def forward(self, images):
        """
        A batch of images (patches)
        """

        # 1. encode 
        d = self.encode(images)

        # 2. crop conv maps output 0 and 1 
        d[0] = crop_op(d[0], (92, 92))
        d[1] = crop_op(d[1], (36, 36))

        # 3. decode
        np_feat = self.decode_np(d)
        npx = self.preact_out_np(np_feat[-1])

        hv_feat = self.decode_hv(d)
        hv = self.preact_out_hv(hv_feat[-1])

        tp_feat = self.decode_tp(d)
        tp = self.preact_out_tp(tp_feat[-1])

        # 4.1 nuclei type pixel (TP)
        logi_class = self.conv_out_tp(tp)
        logi_class = logi_class.permute([0, 2, 3, 1])
        soft_class = F.softmax(logi_class, dim=-1)

        # 4.2 nuclei pixel (NP)
        logi_np = self.conv_out_np(npx)
        logi_np = logi_np.permute([0, 2, 3, 1])
        soft_np = F.softmax(logi_np, dim=-1)
        prob_np = torch.unsqueeze(soft_np[:, :, :, 1], dim=-1)       

        # 4.3 horizontal vertical (HV)
        logi_hv = self.conv_out_hv(hv)
        logi_hv = logi_hv.permute([0, 2, 3, 1])
        prob_hv = logi_hv
        pred_hv = prob_hv

        # 5. concat output 
        predmap_coded = torch.cat([soft_class, prob_np, pred_hv], dim=-1)

        return predmap_coded

class Encoder(nn.Module):

    def __init__(self):
        """

        """
        super(Encoder, self).__init__()
        self.conv0 = Conv2dWithActivation(3, 64, 7, activation='bnrelu', padding=3)  # padding of 3 allows to keep the same dimensions 
        self.group0 = ResidualBlock(64, [64,  64,  256], [1, 3, 1], 3, strides=1)
        self.group1 = ResidualBlock(256, [128, 128,  512], [1, 3, 1], 4, strides=2)
        self.group2 = ResidualBlock(512, [256, 256, 1024], [1, 3, 1], 6, strides=2)
        self.group3 = ResidualBlock(1024, [512, 512, 2048], [1, 3, 1], 3, strides=2)
        self.conv_bot = Conv2dWithActivation(2048, 1024, 1, activation=None)   # @TODO: add 'same' padding 

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.group0(x1)
        x3 = self.group1(x2)
        x4 = self.group2(x3)
        x5 = self.group3(x4)
        x6 = self.conv_bot(x5)
        return [x2, x3, x4, x6]


class ResidualBlock(nn.Module):

    def __init__(self, ch_in, ch, ksize, count, split=1, strides=1):
        """
        
        """
        super(ResidualBlock, self).__init__()
        self.count = count 
        # ch_in = [128, 128]  # @TODO: hardcode the number of input channels in the residual blocks 
        for i in range(0, count):
            if i != 0:
                setattr(self, 'block' + str(i) + '_preact', BNReLU(ch[2])) 
            setattr(self, 'block' + str(i) + '_conv1', Conv2dWithActivation(ch_in if i==0 else ch[-1], ch[0], ksize[0], activation='bnrelu')) 
            setattr(self, 'block' + str(i) + '_conv2', Conv2dWithActivation(ch[0], ch[1], ksize[1], activation='bnrelu', stride=strides if i == 0 else 1, padding=1)) # padding=''same
            setattr(self, 'block' + str(i) + '_conv3', Conv2dWithActivation(ch[1], ch[2], ksize[2], activation=None)) 
            if i == 0:  # (strides != 1 or ch_in[1] != ch[1]) and i == 0:
                setattr(self, 'block' + str(i) + '_convshortcut', Conv2dWithActivation(ch_in, ch[2], 1, stride=strides, activation=None)) 
        self.bnlast = BNReLU(ch[2])  

    def forward(self, l):
        for i in range(0, self.count):

            # set input 
            x = l if i == 0 else getattr(self, 'block' + str(i) + '_preact')(l)

            # loop over conv1 & 2 & 3
            x = getattr(self, 'block' + str(i) + '_conv1')(x)
            x = getattr(self, 'block' + str(i) + '_conv2')(x)
            x = getattr(self, 'block' + str(i) + '_conv3')(x)

            # apply shortcut 
            if hasattr(self, 'block' + str(i) + '_convshortcut'):
                l = getattr(self, 'block' + str(i) + '_convshortcut')(l)
            l = l + x

        # end of each group need an extra activation
        l = self.bnlast(l) 
        return l


class Upsample2x(nn.Module):

    def __init__(self):
        """
        
        """
        super(Upsample2x, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.upsample(x)
        return x


class Decoder(nn.Module):

    def __init__(self):
        """
        
        """
        super(Decoder, self).__init__()

        # variables with name starting by u3
        self.u3_rz = Upsample2x()   
        self.u3_conva = Conv2dWithActivation(1024, 256, 3, activation=None)  
        self.u3_dense = DenseBlock(256, [128, 32], [1, 3], 8, split=4)
        self.u3_convf = Conv2dWithActivation(512, 512, 3, activation=None, padding=1)  # @TODO: define padding 

        # variables with name starting by u2
        self.u2_rz = Upsample2x()
        self.u2_conva = Conv2dWithActivation(512, 256, 3, activation=None)  
        self.u2_dense = DenseBlock(256, [128, 32], [1, 3], 4, split=4)
        self.u2_convf = Conv2dWithActivation(384, 256, 3, activation=None, padding=1) 
     
        # variables with name starting by u1
        self.u1_rz = Upsample2x()
        self.u1_conva = Conv2dWithActivation(256, 64, 3, activation=None)  

    def forward(self, x):

        # processing u3
        u3 = self.u3_rz(x[-1])
        u3_sum = u3 + x[-2]
        u3 = self.u3_conva(u3_sum)
        u3 = self.u3_dense(u3)
        u3 = self.u3_convf(u3)

        # processing u2
        u2 = self.u2_rz(u3)
        u2_sum = u2 + x[-3]
        u2x = self.u2_conva(u2_sum)
        u2 = self.u2_dense(u2x)
        u2 = self.u2_convf(u2)

        # processing u1
        u1 = self.u1_rz(u2)
        u1_sum = u1 + x[-4]
        u1 = self.u1_conva(u1_sum)

        return [u3, u2x, u1]


class DenseBlock(nn.Module):

    def __init__(self, ch_in, ch, ksize, count, split=1):
        """
        
        """
        super(DenseBlock, self).__init__()
        self.count = count
        for i in range(0, count):
            setattr(self, 'blk_' + str(i) + 'preact_bna', BNReLU(ch_in))
            setattr(self, 'blk_' + str(i) + 'conv1', Conv2dWithActivation(ch_in, ch[0], ksize[0], activation='bnrelu'))
            setattr(self, 'blk_' + str(i) + 'conv2', Conv2dWithActivation(ch[0], ch[1], ksize[1], activation=None))
            ch_in = ch_in + ch[-1]

        self.blk_bna = BNReLU(ch_in)

    def forward(self, l):
        for i in range(0, self.count):
            x = getattr(self, 'blk_' + str(i) + 'preact_bna')(l)
            x = getattr(self, 'blk_' + str(i) + 'conv1')(x)
            x = getattr(self, 'blk_' + str(i) + 'conv2')(x)
            x_shape = list(x.shape)
            l_shape = list(l.shape)
            l = crop_op(l, (l_shape[2] - x_shape[2], 
                            l_shape[3] - x_shape[3]))
            l = torch.cat([l, x], dim=1)
        l = self.blk_bna(l)
        return l 

class BNReLU(nn.Module):

    def __init__(self, num_features):
        """
        
        """
        super(BNReLU, self).__init__()
        self.bn =  nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2dWithActivation(nn.Module):

    def __init__(self, num_input, num_output, filter_size, stride=1, activation=None, padding=0, bias=False):
        """
        
        """
        super(Conv2dWithActivation, self).__init__()
        self.conv = nn.Conv2d(num_input, num_output, filter_size, stride=stride, padding=padding, bias=bias)
        self.activation = activation

        if self.activation is not None:
            if activation == 'bnrelu':
                self.act = BNReLU(num_output)
            else:
                raise ValueError('Not implemented')

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.act(x)
        return x 

############# A set of utilities 

def crop_op(x, cropping, data_format='channels_first'):
    """
    Center crop image
    Args:
        cropping is the substracted portion
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == 'channels_first':
        x = x[:,:,crop_t:-crop_b,crop_l:-crop_r]
    else:
        x = x[:,crop_t:-crop_b,crop_l:-crop_r]
    return x   