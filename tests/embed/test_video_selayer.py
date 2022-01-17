from math import log

import torch

from kale.embed.video_selayer import CBAM, CBAMVideo, FCANet, SRM, SRMVideo, STAM, ECANet, ECANetVideo

# Dummy data: [batch_size, channel, time, height, width]
INPUT_5D = torch.randn(2, 64, 16, 32, 32)
# Dummy data: [batch_size, channel, height, width]
INPUT_4D = torch.randn(2, 64, 32, 32)


def test_cbam():
    layer = CBAM(INPUT_4D.shape[1], 2)
    output = layer(INPUT_4D)
    assert output.shape == (2, 64, 32, 32)


def test_cbam_video():
    layer = CBAMVideo(INPUT_5D.shape[1], 2)
    output = layer(INPUT_5D)
    assert output.shape == (2, 64, 16, 32, 32)


def test_stam():
    layer = STAM(INPUT_5D.shape[1], 2)
    output = layer(INPUT_5D)
    assert output.shape == (2, 64, 16, 32, 32)


def test_srm():
    layer = SRM(INPUT_4D.shape[1])
    output = layer(INPUT_4D)
    assert output.shape == (2, 64, 32, 32)


def test_srm_video():
    layer = SRMVideo(INPUT_5D.shape[1])
    output = layer(INPUT_5D)
    assert output.shape == (2, 64, 16, 32, 32)


def test_fcanet():
    layer = FCANet(INPUT_4D.shape[1], 16, 16)
    output = layer(INPUT_4D)
    assert output.shape == (2, 64, 32, 32)


def test_ecanet():
    C = INPUT_4D.shape[1]
    b = 1
    gamma = 2
    t = int(abs((log(C, 2) + b) / gamma))
    k = t if t % 2 else t + 1
    layer = ECANet(k)
    output = layer(INPUT_4D)
    assert output.shape == (2, 64, 32, 32)


def test_ecanet_video():
    C = INPUT_5D.shape[1]
    b = 1
    gamma = 2
    t = int(abs((log(C, 2) + b) / gamma))
    k = t if t % 2 else t + 1
    layer = ECANetVideo(k)
    output = layer(INPUT_5D)
    assert output.shape == (2, 64, 16, 32, 32)
