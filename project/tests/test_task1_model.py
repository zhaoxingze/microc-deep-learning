from __future__ import annotations

import torch

from src.task1.model import SmallMicroCCNN


def test_cnn_maps_batch_one_channel_64_square_to_three_logits() -> None:
    model = SmallMicroCCNN()
    logits = model(torch.zeros(4, 1, 64, 64))
    assert logits.shape == (4, 3)


def test_target_layer_is_last_convolution() -> None:
    model = SmallMicroCCNN()
    convolution_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    assert model.target_layer is convolution_layers[-1]
