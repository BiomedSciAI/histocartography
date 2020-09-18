import torch
import torch.nn as nn
import torch.nn.functional as F 
import dgl


class HoverNet(nn.Module):

    def __init__(self):
        """

        """

        super(HoverNet, self).__init__()

        # define encoder 
        self.encode = Encoder()

        # define decoder(s)
        self.decode_np = Decoder()
        self.decode_hv = Decoder()
        self.decode_tp = Decoder()

        # nuclei type pixel (TP)
        self.conv_out_tp = Conv2dWithActivation(self.nr_types, self.nr_types, 1, activation=None)
        self.preact_out_np = BNReLU(self.nr_types)

        # nuclei pixels (NP)
        self.conv_out_np = Conv2dWithActivation(2, 2, 1, activation=None)
        self.preact_out_hv = BNReLU(2)

        # horizontal-vertival (HV)
        self.conv_out_hv = Conv2dWithActivation(2, 2, 1, activation=None)
        self.preact_out_tp = BNReLU(2)
 
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
        np_feat = self.decode_np('np', d)  
        npx = self.preact_out_np(np_feat[-1])

        hv_feat = self.decode_hv('hv', d)
        hv = self.preact_out_hv(hv_feat[-1])

        tp_feat = self.decode_tp('tp', d)
        tp = self.preact_out_tp(tp_feat[-1])

        # 4.1 nuclei type pixel (TP)
        logi_class = self.conv_out_tp(tp)
        logi_class = logi_class.permute([0, 2, 3, 1])
        soft_class = F.softmax(logi_class, dim=-1)

        # 4.2 nuclei pixel (NP)
        logi_np = self.conv_out_np(npx)
        logi_np = logi_np.permute([0, 2, 3, 1])
        soft_np = F.softmax(logi_np, dim=-1)
        prob_np = soft_np

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
        self.conv0 = Conv2dWithActivation(3, 64, 7, activation=None)
        self.group0 = ResidualBlock([64,  64,  256], [1, 3, 1], 3, strides=1)
        self.group1 = ResidualBlock([128, 128,  512], [1, 3, 1], 4, strides=2)
        self.group2 = ResidualBlock([256, 256, 1024], [1, 3, 1], 6, strides=2)
        self.group3 = ResidualBlock([512, 512, 2048], [1, 3, 1], 3, strides=2)
        self.conv_bot = Conv2dWithActivation(3, 1024, 1, activation=None)  

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.group0(x1)
        x3 = self.group1(x2)
        x4 = self.group2(x3)
        x5 = self.group3(x4)
        x6 = self.conv_bot(x5)
        return [x2, x3, x4, x6]


class Decoder(nn.Module):

    def __init__(self):
        """
        
        """
        super(Decoder, self).__init__()

        # variables with name starting by u3
        self.u3_rz = Upsample2x()  # @TODO: define upsample 
        self.u3_conva = Conv2dWithActivation(256, 256, 3, activation=None)  
        self.u3_dense = DenseBlock([128, 32], [1, 3], 8, split=4)
        self.u3_convf = Conv2dWithActivation(512, 512, 3, activation=None) 

        # variables with name starting by u2
        self.u2_rz = Upsample2x()
        self.u2_conva = Conv2dWithActivation(256, 256, 3, activation=None)  
        self.u2_dense = DenseBlock([128, 32], [1, 3], 8, split=4)
        self.u2_convf = Conv2dWithActivation(512, 512, 3, activation=None) 
     
        # variables with name starting by u1
        self.u1_rz = Upsample2x()
        self.u2_conva = Conv2dWithActivation(64, 64, 3, activation=None)  

    def forward(self, x):
        # define series of operation 
        return x 

class ResidualBlock(nn.Module):

    def __init__(self, ch, ksize, count, split=1, strides=1):
        """
        
        """
        super(ResidualBlock, self).__init__()
        self.count = count 
        ch_in = [128, 128]  # @TODO: hardcode the number of input channel in the residual blocks 
        for i in range(0, count):
            if i != 0:
                self.preact = BNReLU(ch[2])
            self.setattr(self, Conv2dWithActivation(ch[0], ch[0], ksize[0], activation='bnrelu'), 'block' + str(i) + '_conv1') 
            self.setattr(self, Conv2dWithActivation(ch[0], ch[1], ksize[1], activation='bnrelu', stride=strides if i == 0 else 1), 'block' + str(i) + '_conv2') 
            self.setattr(self, Conv2dWithActivation(ch[1], ch[2], ksize[2], activation=None), 'block' + str(i) + '_conv3') 
            if (strides != 1 or ch_in[1] != ch[2]) and i == 0:
                self.setattr(self, Conv2dWithActivation(ch[2], ch[2], 1, stride=strides, activation=None), 'block' + str(i) + '_convshortcut') 
        self.bnlast = BNReLU(ch[2])  

    def forward(self, l):
        for i in range(0, self.count):

            # set input 
            x = l if i == 0 else self.preact(l)

            # loop over conv1 & 2 & 3
            x = getattr(self, 'block' + str(i) + '_conv1')(x)
            x = getattr(self, 'block' + str(i) + '_conv2')(x)
            x = getattr(self, 'block' + str(i) + '_conv3')(x)

            # apply shortcut 
            if hasattr(self, 'block' + str(i) + '_convshortcut'):
                l = getattr(self, 'block' + str(i) + '.convshortcut')(l)
            l = l + x

        # end of each group need an extra activation
        l = self.bnlast(l) 
        return l


class DenseBlock(nn.Module):

    def __init__(self, l, ch, ksize, count, split=1):
        """
        
        """
        super(DenseBlock, self).__init__()
        self.count = count
        for i in range(0, count):
            # with tf.variable_scope('blk/' + str(i)):

            self.preact_bna = BNReLU(ch[0])
            setattr(self, Conv2dWithActivation(ch[0], ch[0], ksize[0], activation='bnrelu'), 'blk_' + str(i) + 'conv1')
            setattr(self, Conv2dWithActivation(ch[0], ch[1], ksize[1], activation=None), 'blk_' + str(i) + 'conv2')

        self.blk_bna = BNReLU(ch[1])

    def forward(self, l):
        for i in range(0, self.count):
            x = self.preact_bna(l)
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
        return 


class Conv2dWithActivation(nn.Module):

    def __init__(self, num_input, num_output, filter_size, stride=1, activation=None):
        """
        
        """
        super(Conv2dWithActivation, self).__init__()
        self.conv = nn.Conv2d(num_input, num_output, filter_size, stride=stride)
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
        return 

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