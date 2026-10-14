import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("pytorch_metric_learning")
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

from a2b_lightvpr.vpr.metric import metric_place_loss


def _official(descriptors, labels):
    value = descriptors.float()
    miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
    objective = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    pairs = miner(value, labels)
    return objective(value, labels, pairs), pairs, objective


def test_metric_matches_official_multisimilarity_forward_and_gradients():
    torch.manual_seed(12)
    value = F.normalize(torch.randn(16, 12), dim=1)
    descriptors, reference = value.clone().requires_grad_(), value.clone().requires_grad_()
    labels = torch.arange(4).repeat_interleave(4)
    loss, components = metric_place_loss(descriptors, labels)
    expected, pairs, objective = _official(reference, labels)
    assert loss > 0
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(descriptors.grad, reference.grad, rtol=0, atol=0)
    assert components["mined_positive_pairs"].item() == len(pairs[0])
    assert components["mined_negative_pairs"].item() == len(pairs[2])
    assert len(pairs[0]) > len(labels)  # More than one positive per anchor.
    per_anchor = objective.compute_loss(reference.detach(), labels, pairs, reference.detach(), labels)["loss"]["losses"]
    active = per_anchor.reshape(-1).ne(0)
    positive, negative = torch.zeros_like(active), torch.zeros_like(active)
    positive[pairs[0]], negative[pairs[2]] = True, True
    torch.testing.assert_close(components["active_anchor_fraction"], active.float().mean())
    torch.testing.assert_close(components["both_pair_anchor_fraction"], (active & positive & negative).float().mean())
    for value in components.values():
        assert value.ndim == 0 and torch.isfinite(value) and not value.requires_grad


@pytest.mark.parametrize("labels", [
    torch.zeros(4, dtype=torch.long),  # No negative place.
    torch.arange(4),                 # No positive place.
    torch.tensor([0]),              # Singleton batch.
    torch.empty(0, dtype=torch.long),
])
def test_no_pairs_return_finite_graph_connected_zero(labels):
    descriptors = torch.randn(len(labels), 8, requires_grad=True)
    loss, components = metric_place_loss(descriptors, labels)
    assert loss.item() == 0 and loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert descriptors.grad is not None
    assert torch.count_nonzero(descriptors.grad) == 0
    assert all(value.item() == 0 for value in components.values())


def test_easy_perfect_places_have_no_mined_pairs():
    descriptors = torch.eye(3).repeat_interleave(3, dim=0).requires_grad_()
    labels = torch.arange(3).repeat_interleave(3)
    loss, components = metric_place_loss(descriptors, labels)
    expected, pairs, _ = _official(descriptors, labels)
    assert all(index.numel() == 0 for index in pairs)
    torch.testing.assert_close(loss, expected)
    assert components["active_anchor_fraction"] == 0
    loss.backward()
    assert torch.isfinite(descriptors.grad).all()


def test_cpu_amp_objective_runs_in_fp32_with_official_gradients():
    torch.manual_seed(10)
    value = F.normalize(torch.randn(12, 10), dim=1).to(torch.bfloat16)
    descriptors, reference = value.clone().requires_grad_(), value.clone().requires_grad_()
    labels = torch.arange(3).repeat_interleave(4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, _ = metric_place_loss(descriptors, labels)
    expected, _, _ = _official(reference, labels)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(descriptors.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP requires a GPU")
def test_cuda_amp_objective_is_finite_and_matches_official():
    torch.manual_seed(10)
    value = F.normalize(torch.randn(12, 10, device="cuda"), dim=1).half()
    descriptors, reference = value.clone().requires_grad_(), value.clone().requires_grad_()
    labels = torch.arange(3, device="cuda").repeat_interleave(4)
    with torch.autocast("cuda", dtype=torch.float16):
        actual, _ = metric_place_loss(descriptors, labels)
    expected, _, _ = _official(reference, labels)
    assert actual.dtype == torch.float32 and torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(descriptors.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("descriptors,labels", [
    (torch.randn(3, 4), torch.tensor([0, 1])),
    (torch.randn(3, 4), torch.tensor([[0], [1], [1]])),
    (torch.randn(3, 4), torch.tensor([0., 1., 1.])),
    (torch.randn(3, 4), torch.tensor([True, False, False])),
    (torch.randn(3, 4, 1), torch.tensor([0, 1, 1])),
    (torch.empty(3, 0), torch.tensor([0, 1, 1])),
])
def test_invalid_metric_metadata_fails(descriptors, labels):
    with pytest.raises(ValueError):
        metric_place_loss(descriptors, labels)


def test_active_anchor_stats_observe_pml_zero_pair_shortcut(monkeypatch):
    # PML currently reduces a <=1-pair batch directly to zero; observing its
    # actual loss vector prevents reporting these merely mined anchors active.
    pair = torch.tensor([0])
    monkeypatch.setattr(miners.MultiSimilarityMiner, "forward", lambda *args: (pair, pair + 1, pair + 1, pair + 2))
    descriptors = torch.randn(3, 8, requires_grad=True)
    loss, components = metric_place_loss(descriptors, torch.tensor([0, 0, 1]))
    assert loss.item() == 0
    assert components["mined_positive_pairs"] == components["mined_negative_pairs"] == 1
    assert components["mined_anchor_fraction"] == pytest.approx(2 / 3)
    assert components["active_anchor_fraction"] == components["both_pair_anchor_fraction"] == 0
    loss.backward()
    assert descriptors.grad is not None
