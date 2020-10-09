import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BiDerainNet(nn.Module):
    def __init__(self, n_channels, heatmap_num, feature_only):
        super(BiDerainNet, self).__init__()

        self.texture_path = UNet(n_channels=64,
                                 heatmap_num=heatmap_num,
                                 feature_only=False)

        self.feature_only = feature_only
        self.inc = inconv(n_channels, 64)
        self.fusion_block = FeatureHeatmapFusingBlock(n_channels, 1, 3)
        self.outc = outconv(64, n_channels)

    def forward(self, x, heatmap_feature=None, edge_feature=None):

        # class texture path
        texture = self.texture_path(x, heatmap_feature)

        # sharp_edge path
        edge = self.inc(x)
        edge = self.fusion_block(edge, edge_feature)
        edge = self.outc(edge)

        # merge two path
        return torch.add(texture, edge)


class BiDerainNet_first(nn.Module):
    def __init__(self, n_channels):
        super(BiDerainNet_first, self).__init__()

        self.texture_path = UNet_first(n_channels=64)

        self.inc = inconv(n_channels, 64)
        self.outc = outconv(64, n_channels)

    def forward(self, x):
        # class texture path
        texture = self.texture_path(x)

        # sharp_edge path
        edge = self.inc(x)
        edge = self.outc(edge)

        # merge two path
        derain = torch.add(texture, edge)

        return derain


class UNet(nn.Module):
    def __init__(self, n_channels, heatmap_num, feature_only):
        super(UNet, self).__init__()
        self.feature_only = feature_only
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 64)
        self.down2 = down(64, 64)
        self.down3 = down(64, 64)
        if not feature_only:
            self.up1 = up(128, 64)
            self.up2 = up(128, 64)
            self.up3 = up(128, 64)
            self.outc = outconv(64, n_channels)
        self.fusion_block = FeatureHeatmapFusingBlock(n_channels,
                                                      heatmap_num,
                                                      3)

    def forward(self, x, feature=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if feature is not None:
            x4 = self.fusion_block(x4, feature)
        if not self.feature_only:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.outc(x)
            # return 3 scale features and reconstructed picture
            # return (x2, x3, x4), x
            return x
        else:
            return (x2, x3, x4), None


class UNet_first(nn.Module):
    def __init__(self, n_channels):
        super(UNet_first, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 64)
        self.down2 = down(64, 64)
        self.down3 = down(64, 64)
        self.up1 = up(128, 64)
        self.up2 = up(128, 64)
        self.up3 = up(128, 64)
        self.outc = outconv(64, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x


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


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.AvgPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FeatureHeatmapFusingBlock(nn.Module):
    def __init__(self,
                 feat_channel_in,
                 num_heatmap,
                 num_block,
                 num_mid_channel=None):
        super().__init__()
        self.num_heatmap = num_heatmap
        res_block_channel = feat_channel_in * num_heatmap
        if num_mid_channel is None:
            self.num_mid_channel = num_heatmap * feat_channel_in
        else:
            self.num_mid_channel = num_mid_channel
        self.conv_in = ConvBlock(feat_channel_in, res_block_channel, 1, norm_type=None, act_type='lrelu')
        self.resnet = nn.Sequential(*[
            ResBlock(res_block_channel,
                     res_block_channel,
                     self.num_mid_channel,
                     3,
                     norm_type=None,
                     act_type='lrelu',
                     groups=num_heatmap) for _ in range(num_block)
        ])

    def forward(self, feature, heatmap, debug=False):
        assert self.num_heatmap == heatmap.size(1)
        batch_size = heatmap.size(0)
        w, h = feature.shape[-2:]

        heatmap = F.interpolate(heatmap, [feature.size(2), feature.size(3)], mode='bilinear', align_corners=True)

        attention = nn.functional.softmax(heatmap, dim=1)  # B * num_heatmap * h * w
        feature = self.conv_in(feature)
        feature = self.resnet(feature)  # B * (num_heatmap*feat_channel_in) * h * w

        if debug:
            feature = feature.view(batch_size, self.num_heatmap, -1, w, h)
            return feature, attention.unsqueeze(2)
        else:
            feature = feature.view(batch_size, self.num_heatmap, -1, w, h) * attention.unsqueeze(2)
            feature = feature.sum(1)
            return feature


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, kernel_size, stride=1, valid_padding=True, padding=0, dilation=1, bias=True, \
                 pad_type='zero', norm_type='bn', act_type='relu', mode='CNA', res_scale=1, groups=1):
        super(ResBlock, self).__init__()
        conv0 = ConvBlock(in_channel, mid_channel, kernel_size, stride, dilation, bias, valid_padding, padding, act_type, norm_type, pad_type, mode, groups=groups)
        act_type = None
        norm_type = None
        conv1 = ConvBlock(mid_channel, out_channel, kernel_size, stride, dilation, bias, valid_padding, padding, act_type, norm_type, pad_type, mode, groups=groups)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,\
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA', groups=1):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

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
