import torch
import torch.nn as nn
import sys
from collections import OrderedDict
import torch.nn.functional as F


class DIC(nn.Module):
    def __init__(self, args, derain_criterion=nn.MSELoss, seg_criterion=nn.CrossEntropyLoss, is_train=True):
        super().__init__()
        in_channels = 3
        out_channels = 3
        act_type = 'prelu'
        norm_type = 'bn'
        self.is_train = is_train
        self.derain_criterion = derain_criterion
        self.seg_criterion = seg_criterion

        self.num_steps = args.num_steps
        num_features = args.conv_block_num_features

        # LR feature extraction block
        self.conv_in = ConvBlock(
            in_channels,
            num_features,
            kernel_size=3,
            act_type=act_type,
            norm_type=norm_type)

        if args.derain_arch == 'unet':
            from .modules.derain_unet import UNet_first, UNet
            self.first_block = UNet_first(n_channels=64)
            self.block = UNet(n_channels=64,
                              heatmap_num=args.classes,
                              feature_only=False)
        elif args.derain_arch == 'BiDerainNet':
            from .modules.BiDerainNet import BiDerainNet_first, BiDerainNet
            self.first_block = BiDerainNet_first(n_channels=64)
            self.block = BiDerainNet(n_channels=64,
                                     heatmap_num=args.classes,
                                     feature_only=False)
        else:
            raise RuntimeError('Undefined Net Type for Derain.')

        self.conv_out = ConvBlock(
            num_features,
            num_features,
            kernel_size=3,
            act_type=None,
            norm_type=norm_type)

        self.edge_net = edge_net(class_num=args.classes)

        self.derain_final_conv = nn.Conv2d(num_features, out_channels,
                                           kernel_size=3, stride=1,
                                           padding=1)

        if args.seg_arch == 'bisenetr18':
            from .modules.bisenet_r18 import BiSeNet
            if is_train:
                self.seg_net = BiSeNet(out_planes=args.classes,
                                       norm_layer=norm_type,
                                       output_size=[args.train_h, args.train_w])
            else:
                self.seg_net = BiSeNet(out_planes=args.classes,
                                       norm_layer=norm_type,
                                       output_size=[args.test_h, args.test_w])
        elif args.seg_arch == 'bisenetx39':
            from .modules.bisenet_x39 import BiSeNet
            if is_train:
                self.seg_net = BiSeNet(out_planes=args.classes,
                                       norm_layer=norm_type,
                                       output_size=[args.train_h, args.train_w])
            else:
                self.seg_net = BiSeNet(out_planes=args.classes,
                                       norm_layer=norm_type,
                                       output_size=[args.test_h, args.test_w])
        else:
            raise RuntimeError('Undefined Net Type for Segmentation.')

    def forward(self, x, clear_label=None, seg_label=None):

        inter_res = x

        x = self.conv_in(x)
        derain_outs = []
        seg_outs = []
        heatmap = None
        edge_map = None

        for step in range(self.num_steps):
            if step == 0:
                rain_residual = self.derain_final_conv(self.conv_out(self.first_block(x)))
                derain = torch.add(inter_res, rain_residual)
                heatmap = self.seg_net(derain)
                edge_map = self.edge_net(heatmap)
            else:
                rain_residual = self.derain_final_conv(self.conv_out(self.block(x, merge_heatmap(heatmap), edge_map)))
                derain = torch.add(inter_res, rain_residual)
                heatmap = self.seg_net(derain)
                edge_map = self.edge_net(heatmap)

            derain_outs.append(derain)
            seg_outs.append(heatmap)

        if self.is_train:
            derain_losses = []
            seg_losses = []
            for derain_out in derain_outs:
                derain_losses.append(self.derain_criterion(derain_out, clear_label))
            for seg_out in seg_outs:
                seg_losses.append(self.seg_criterion(seg_out, seg_label))
            return derain_outs[-1], seg_outs[-1].max(1)[1], derain_losses, seg_losses
        else:
            return derain_outs, seg_outs


class edge_net(nn.Module):
    def __init__(self, class_num, mid_ch=64, out_ch=1):
        super(edge_net, self).__init__()
        self.in_conv = double_conv(class_num, mid_ch)
        self.mid_conv = double_conv(mid_ch, mid_ch)
        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1)
        )

    def forward(self, x):
        return self.out_conv(self.mid_conv(self.in_conv(x)))


class double_conv(nn.Module):
    '''(conv => LeakyReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def merge_heatmap(heatmap_in):
    '''
    merge 68 heatmap to 5
    heatmap: B*N*32*32
    '''
    # landmark[36:42], landmark[42:48], landmark[27:36], landmark[48:68]
    heatmap = heatmap_in.clone()
    max_heat = heatmap.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    max_heat = torch.max(max_heat, torch.ones_like(max_heat) * 0.05)
    heatmap /= max_heat
    if heatmap.size(1) == 19:
        return heatmap
    else:
        raise NotImplementedError('Fusion for face landmark number %d not implemented!' % heatmap.size(1))


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,\
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA', groups=1):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'% sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)


def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
    padding = (kernel_size-1) // 2
    return padding


def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!'%act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type =='bn':
        layer = nn.BatchNorm2d(n_feature)
    elif norm_type =='sync_bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!'%norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!'%pad_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict'%sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)
