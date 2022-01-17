# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
# =============================================================================

"""Python implementation of variable attention modules, e.g. Squeeze-and-Excitation Layers (SELayer)
Initial implementation: channel-wise (SELayerC)
Improved implementation: temporal-wise (SELayerT), convolution-based channel-wise (SELayerCoC), max-pooling-based
channel-wise (SELayerMC), multi-pooling-based channel-wise (SELayerMAC)

[Redundancy and repeat of code will be reduced in the future.]
"""
import math

import torch
import torch.nn as nn
from torch.nn import Parameter


def get_attention(attention):
    """Get attention modules.

    Args:
        attention (string): the name of the attention module.
            (Options: ["SELayerC", "SELayerT", "SRMVideo", "CBAMVideo", "STAM",
            "SELayerCoC", "SELayerMC", "SELayerMAC"])

    Returns:
        module (nn.Module, optional): the attention module.
    """

    if attention == "SELayerC":
        module = SELayerC
    elif attention == "SELayerT":
        module = SELayerT
    elif attention == "SRMVideo":
        module = SRMVideo
    elif attention == "CBAMVideo":
        module = CBAMVideo
    elif attention == "STAM":
        module = STAM
    elif attention == "SELayerCoC":
        module = SELayerCoC
    elif attention == "SELayerMC":
        module = SELayerMC
    elif attention == "SELayerMAC":
        module = SELayerMAC
    elif attention == "SELayerCoC":
        module = SELayerCoC

    else:
        raise ValueError("Wrong MODEL.ATTENTION. Current:{}".format(attention))
    return module


class SELayer(nn.Module):
    """Helper class for SELayer design.

    References:
    Hu Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." In CVPR, pp. 7132-7141. 2018.
    For initial implementation, please go to https://github.com/hujie-frank/SENet
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.channel = channel
        self.reduction = reduction

    def forward(self, x):
        return NotImplementedError()


class SELayerC(SELayer):
    """Construct channel-wise SELayer."""

    def __init__(self, channel, reduction=16):
        super(SELayerC, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerT(SELayer):
    """Construct temporal-wise SELayer."""

    def __init__(self, channel, reduction=2):
        super(SELayerT, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, _, t, _, _ = x.size()
        output = x.transpose(1, 2).contiguous()
        y = self.avg_pool(output).view(b, t)
        y = self.fc(y).view(b, t, 1, 1, 1)
        y = y.transpose(1, 2).contiguous()
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SRM(nn.Module):
    """Construct Style-based Recalibration Module for images.

    References:
        Lee, HyunJae, Hyo-Eun Kim, and Hyeonseob Nam. "Srm: A style-based recalibration module for convolutional neural
        networks." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1854-1862. 2019.

        https://github.com/hyunjaelee410/style-based-recalibration-module/blob/master/models/recalibration_modules.py
    """

    def __init__(self, channel):
        super(SRM, self).__init__()
        self.channel = channel

        # CFC: channel-wise fully connected layer
        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, "srm_param", True)
        setattr(self.bn.weight, "srm_param", True)
        setattr(self.bn.bias, "srm_param", True)

    def forward(self, x, eps=1e-5):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(dim=2, keepdim=True)
        var = x.view(b, c, -1).var(dim=2, keepdim=True) + eps
        std = var.sqrt()

        t = torch.cat((mean, std), dim=2)  # (b, c, 2)

        # Style integration
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)
        out = x * g.expand_as(x)
        return out


class SRMVideo(SRM):
    """This module extends SRM for videos."""

    def __init__(self, channel):
        super(SRMVideo, self).__init__(channel)
        self.bn = nn.BatchNorm3d(self.channel)

    def forward(self, x, eps=1e-5):
        b, c, _, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(dim=2, keepdim=True)
        var = x.view(b, c, -1).var(dim=2, keepdim=True) + eps
        std = var.sqrt()

        t = torch.cat((mean, std), dim=2)  # (b, c, 2)

        # Style integration
        z = t * self.cfc[None, :, :]  # b x c x 2
        z = torch.sum(z, dim=2)[:, :, None, None, None]  # b x c x 1 x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        g = g - 0.5
        out = x + x * g.expand_as(x)
        # out = x * g.expand_as(x)
        return out


class CBAM(nn.Module):
    """Construct Convolutional Block Attention Module.

    References:
        [1] Woo, Sanghyun, Jongchan Park, Joon-Young Lee, and In So Kweon. "Cbam: Convolutional block attention
        module." In Proceedings of the European conference on computer vision (ECCV), pp. 3-19. 2018.
    """

    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.CAM = CBAMChannelModule(channel, reduction)
        self.SAM = CBAMSpatialModule()

    def forward(self, x):
        y = self.CAM(x)
        y = self.SAM(y)
        return y


class CBAMChannelModule(SELayer):
    def __init__(self, channel, reduction=16):
        super(CBAMChannelModule, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1)
        y = torch.add(y_avg, y_max)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class CBAMSpatialModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(CBAMSpatialModule, self).__init__()
        self.kernel_size = kernel_size
        self.compress = CBAMChannelPool()
        self.conv = nn.Conv2d(2, 1, self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        y = self.conv(x_compress)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class CBAMVideo(nn.Module):
    """This module extends CBAM for videos by applying 3D layers."""

    def __init__(self, channel, reduction=16):
        super(CBAMVideo, self).__init__()
        self.CAM = CBAMChannelModuleVideo(channel, reduction)
        self.SAM = CBAMSpatialModuleVideo()

    def forward(self, x):
        y = self.CAM(x)
        y = self.SAM(y)
        return y


class CBAMChannelModuleVideo(CBAMChannelModule):
    def __init__(self, channel, reduction=16):
        super(CBAMChannelModuleVideo, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1, 1)
        y = torch.add(y_avg, y_max)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class CBAMSpatialModuleVideo(CBAMSpatialModule):
    def __init__(self, kernel_size=7):
        super(CBAMSpatialModuleVideo, self).__init__(kernel_size)
        self.conv = nn.Conv3d(2, 1, self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        y = self.conv(x_compress)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class CBAMChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class FCANet(nn.Module):
    """Construct Frequency Channel Attention Module.

    References:
        Qin, Zequn, Pengyi Zhang, Fei Wu, and Xi Li. "Fcanet: Frequency channel attention networks."
        In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 783-792. 2021.

        https://github.com/cfzd/FcaNet/blob/master/model/layer.py
    """

    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method="top16"):
        super(FCANet, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = self.get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)

    def get_freq_indices(self, method):
        assert method in [
            "top1",
            "top2",
            "top4",
            "top8",
            "top16",
            "top32",
            "bot1",
            "bot2",
            "bot4",
            "bot8",
            "bot16",
            "bot32",
            "low1",
            "low2",
            "low4",
            "low8",
            "low16",
            "low32",
        ]
        num_freq = int(method[3:])
        if "top" in method:
            top_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
            top_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
            mapper_x = top_x[:num_freq]
            mapper_y = top_y[:num_freq]
        elif "low" in method:
            low_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
            low_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
            mapper_x = low_x[:num_freq]
            mapper_y = low_y[:num_freq]
        elif "bot" in method:
            bot_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
            bot_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
            mapper_x = bot_x[:num_freq]
            mapper_y = bot_y[:num_freq]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y


class MultiSpectralDCTLayer(nn.Module):
    """Generate dct filters for FCANet."""

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer("weight", self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, "x must been 4 dimensions, but got " + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(
                        t_x, u_x, tile_size_x
                    ) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter


class ECANet(nn.Module):
    """Constructs Efficient Channel Attention Module.

    Args:
        kernel_size: Adaptive selection of kernel size

    References:
        Wang, Qilong and Wu, Banggu and Zhu, Pengfei and Li, Peihua and Zuo, Wangmeng and Hu, Qinghua. "ECA-Net:
        Efficient Channel Attention for Deep Convolutional Neural Networks." In Proceedings of the IEEE/CVF Conference
        on Computer Vision and Pattern Recognition (CVPR), pp. 11534-11542. 2020.
        https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py


    """

    def __init__(self, kernel_size=3):
        super(ECANet, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  # batch x channel x height x weight

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECANetVideo(ECANet):
    """This module extends ECANet for videos."""

    def __init__(self, kernel_size=3):
        super(ECANetVideo, self).__init__(kernel_size)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)


class STAM(SELayer):
    """Construct Spatial-temporal Attention Module.

    References:
        Zhou, Shengwei, Liang Bai, Haoran Wang, Zhihong Deng, Xiaoming Zhu, and Cheng Gong. "A Spatial-temporal
        Attention Module for 3D Convolution Network in Action Recognition." DEStech Transactions on Computer
        Science and Engineering cisnrc (2019).
    """

    def __init__(self, channel, reduction=16):
        super(STAM, self).__init__(channel, reduction)
        self.kernel_size = 7
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
        )
        self.conv = nn.Conv3d(1, 1, self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        y = self.sigmoid(y)
        y = x * y.expand_as(x)
        y = y.mean(1).unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y)
        out = x + x * y.expand_as(x)
        return out


class SELayerCoC(SELayer):
    """Construct convolution-based channel-wise SELayer."""

    def __init__(self, channel, reduction=16):
        super(SELayerCoC, self).__init__(channel, reduction)
        self.conv1 = nn.Conv3d(
            in_channels=self.channel, out_channels=self.channel // self.reduction, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(num_features=self.channel // self.reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv3d(
            in_channels=self.channel // self.reduction, out_channels=self.channel, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(num_features=self.channel)

    def forward(self, x):
        b, c, t, _, _ = x.size()  # n, c, t, h, w
        y = self.conv1(x)  # n, c/r, t, h, w
        y = self.bn1(y)  # n, c/r, t, h, w
        y = self.avg_pool(y)  # n, c/r, 1, 1, 1
        y = self.conv2(y)  # n, c, 1, 1, 1
        y = self.bn2(y)  # n, c, 1, 1, 1
        y = self.sigmoid(y)  # n, c, 1, 1, 1
        # out = x * y.expand_as(x)  # n, c, t, h, w
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerMC(SELayer):
    """Construct channel-wise SELayer with max pooling."""

    def __init__(self, channel, reduction=16):
        super(SELayerMC, self).__init__(channel, reduction)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.max_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerMAC(SELayer):
    """Construct channel-wise SELayer with the mix of average pooling and max pooling."""

    def __init__(self, channel, reduction=16):
        super(SELayerMAC, self).__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), bias=False)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = torch.cat((y_avg, y_max), dim=2).squeeze().unsqueeze(dim=1)
        y = self.conv(y).squeeze()
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out
