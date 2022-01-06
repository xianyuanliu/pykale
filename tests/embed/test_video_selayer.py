import torch

from kale.embed.video_selayer import CBAMLayerVideo, SRMLayer, SRMLayerVideo, STAMLayer, CBAMLayer

# Dummy data: [batch_size, time, channel, height, width]
INPUT_5D = torch.randn(2, 4, 8, 16, 16)
INPUT_4D = torch.randn(2, 4, 16, 16)


def test_cbam_layer():
    layer = CBAMLayer(INPUT_5D.shape[1], 2)
    output = layer(INPUT_4D)
    assert output.shape == (2, 4, 16, 16)


def test_cbam_video_layer():
    layer = CBAMLayerVideo(INPUT_5D.shape[1], 2)
    output = layer(INPUT_5D)
    assert output.shape == (2, 4, 8, 16, 16)


def test_stam_layer():
    layer = STAMLayer(INPUT_5D.shape[1], 2)
    output = layer(INPUT_5D)
    assert output.shape == (2, 4, 8, 16, 16)


def test_srm_layer():
    layer = SRMLayer(INPUT_4D.shape[1], 2)
    output = layer(INPUT_4D)
    assert output.shape == (2, 4, 16, 16)


def test_srm_video_layer():
    layer = SRMLayerVideo(INPUT_5D.shape[1], 2)
    output = layer(INPUT_5D)
    assert output.shape == (2, 4, 8, 16, 16)
