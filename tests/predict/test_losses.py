import pytest
import torch

from kale.predict.losses import multitask_topk_accuracy, topk_accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dummy data: [batch_size, num_classes]
FIRST_PREDS = torch.tensor(
    (
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
).to(device)
FIRST_LABELS = torch.tensor((0, 2, 4, 5, 5)).to(device)

SECOND_PREDS = torch.tensor(
    (
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
).to(device)
SECOND_LABELS = torch.tensor((0, 0, 4, 4, 5)).to(device)

MULTI_PREDS = (FIRST_PREDS, SECOND_PREDS)
MULTI_LABELS = (FIRST_LABELS, SECOND_LABELS)


def test_topk_accuracy():
    # Test topk_accuracy with single-label input
    preds = FIRST_PREDS
    labels = FIRST_LABELS
    k = (1, 3, 5)

    top1, top3, top5 = topk_accuracy(preds, labels, k)
    top1_value = top1.double().mean()
    top3_value = top3.double().mean()
    top5_value = top5.double().mean()
    assert top1_value.cpu() == pytest.approx(1 / 5)
    assert top3_value.cpu() == pytest.approx(2 / 5)
    assert top5_value.cpu() == pytest.approx(3 / 5)


def test_multitask_topk_accuracy():
    # Test multitask_topk_accuracy with two-label input
    preds = MULTI_PREDS
    labels = MULTI_LABELS
    k = (1, 3, 5)

    top1, top3, top5 = multitask_topk_accuracy(preds, labels, k)
    top1_value = top1.double().mean()
    top3_value = top3.double().mean()
    top5_value = top5.double().mean()
    assert top1_value.cpu() == pytest.approx(1 / 5)
    assert top3_value.cpu() == pytest.approx(2 / 5)
    assert top5_value.cpu() == pytest.approx(3 / 5)
