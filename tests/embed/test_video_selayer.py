import torch

from kale.embed.video_selayer import CBAM, CBAMVideo, SRM, SRMVideo, STAM

# Dummy data: [batch_size, channel, time, height, width]
INPUT_5D = torch.randn(2, 3, 16, 32, 32)
INPUT_4D = torch.randn(2, 3, 32, 32)


def test_cbam_layer():
    layer = CBAM(INPUT_4D.shape[1], 2)
    output = layer(INPUT_4D)
    assert output.shape == (2, 3, 32, 32)


def test_cbam_video_layer():
    layer = CBAMVideo(INPUT_5D.shape[1], 2)
    output = layer(INPUT_5D)
    assert output.shape == (2, 3, 16, 32, 32)


def test_stam_layer():
    layer = STAM(INPUT_5D.shape[1], 2)
    output = layer(INPUT_5D)
    assert output.shape == (2, 3, 16, 32, 32)


def test_srm_layer():
    layer = SRM(INPUT_4D.shape[1])
    output = layer(INPUT_4D)
    assert output.shape == (2, 3, 32, 32)


def test_srm_video_layer():
    layer = SRMVideo(INPUT_5D.shape[1])
    output = layer(INPUT_5D)
    assert output.shape == (2, 3, 16, 32, 32)
