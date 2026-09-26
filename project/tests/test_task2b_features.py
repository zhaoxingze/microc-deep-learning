import numpy as np
import torch

from src.task1.model import SmallMicroCCNN
from src.task2.autoencoder import BackgroundAutoencoder
from src.task2b.features import (
    ae_encoder_embedding,
    average_replicate_blocks,
    block_cosine_similarity,
    cnn_preprocess,
    fuse_blocks,
)


def test_forward_features_preserves_cnn_logits_and_is_preclassifier_hidden():
    torch.manual_seed(2)
    model = SmallMicroCCNN().eval()
    matrix = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        old_formula = model.classifier(model.pool(model.features(matrix)))
        hidden = model.forward_features(matrix)
        new_logits = model(matrix)
    assert hidden.shape == (2, 64)
    assert torch.equal(new_logits, old_formula)
    assert torch.equal(hidden, model.classifier[:3](model.pool(model.features(matrix))))


def test_ae_encoder_adaptive_pooling_has_fixed_dimension_at_all_scales():
    model = BackgroundAutoencoder().eval()
    for bins in (32, 64, 128):
        with torch.no_grad():
            embedding = ae_encoder_embedding(model, torch.ones(2, 1, bins, bins))
        assert embedding.shape == (2, 64)


def test_replicate_average_and_fusion_order_are_fixed():
    one = {"cnn": np.ones((2, 2)), "small": np.full((2, 3), 2),
           "medium": np.full((2, 1), 3), "large": np.full((2, 4), 4)}
    two = {name: values + 2 for name, values in one.items()}
    average = average_replicate_blocks(one, two)
    fused = fuse_blocks(average)
    assert fused.shape == (2, 10)
    assert np.array_equal(fused[0], np.array([2, 2, 3, 3, 3, 4, 5, 5, 5, 5]))


def test_replicate_cosine_identical_and_orthogonal():
    left = np.array([[1., 0.], [1., 0.]])
    right = np.array([[1., 0.], [0., 1.]])
    assert block_cosine_similarity(left, right).tolist() == [1.0, 0.0]


def test_cnn_raw_preprocessing_is_exact_copy_of_selected_raw_matrix():
    raw = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    transformed = cnn_preprocess(raw, "raw")
    assert np.array_equal(transformed, raw)
    assert transformed is not raw
