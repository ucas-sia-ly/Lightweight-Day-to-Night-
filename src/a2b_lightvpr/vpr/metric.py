"""Standard SALAD metric objective from serizba/salad's utils/losses.py.

Batch construction must supply one view per distinct source, multiple source
images per place, and globally unique place labels across datasets. Source
identity cannot be inferred from descriptors or place labels in this interface.
"""
from __future__ import annotations

import torch


def metric_place_loss(
    descriptors: torch.Tensor, labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply official MultiSimilarity mining/loss parameters in FP32.

    Descriptor models already produce normalized embeddings; the PML distances
    retain their official default behavior. ``active_anchor_fraction`` counts
    nonzero per-anchor losses before PML's standard MeanReducer, including
    anchors with only positive or only negative mined pairs. The ``both_pair``
    fraction counts active anchors with both kinds of pair. All fractions use
    the full batch denominator, as does MeanReducer. Zero-pair batches return a
    finite zero connected to the descriptor graph. Components are detached.
    """
    if descriptors.ndim != 2 or descriptors.shape[1] < 1:
        raise ValueError("Metric descriptors must have shape [N,D] with positive D")
    if labels.ndim != 1 or labels.shape[0] != descriptors.shape[0]:
        raise ValueError("Metric place labels must have matching shape [N]")
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise ValueError("Metric place labels must be integer place IDs")
    try:
        from pytorch_metric_learning import losses, miners
        from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
    except ImportError as error:
        raise RuntimeError("Metric training requires pytorch-metric-learning; reinstall project dependencies with: python -m pip install -e .") from error

    miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
    objective = losses.MultiSimilarityLoss(
        alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity(),
    )
    with torch.autocast(device_type=descriptors.device.type, enabled=False):
        embeddings = descriptors.float()
        targets = labels.to(device=embeddings.device)
        positive_anchors, positives, negative_anchors, negatives = miner(embeddings, targets)
        indices = positive_anchors, positives, negative_anchors, negatives
        batch_size = embeddings.shape[0]
        active = torch.zeros(batch_size, dtype=torch.bool, device=embeddings.device)

        def record_active_anchors(module, arguments):
            # Observe the actual PML loss vector; do not recreate its formula,
            # pair-selection shortcuts or reducer behavior for diagnostics.
            info = arguments[0]["loss"]
            values = info["losses"]
            if info["reduction_type"] == "element" and isinstance(values, torch.Tensor):
                active[info["indices"]] = values.detach().reshape(-1).ne(0)

        handle = objective.reducer.register_forward_pre_hook(record_active_anchors)
        try:
            loss = objective(embeddings, targets, indices)
        finally:
            handle.remove()
        if not any(index.numel() for index in indices):
            loss = embeddings.sum() * 0.0

        positive_mask, negative_mask = torch.zeros_like(active), torch.zeros_like(active)
        positive_mask[positive_anchors] = True
        negative_mask[negative_anchors] = True
        denominator = max(1, batch_size)
        components = {
            "place": loss.detach(),
            "mined_positive_pairs": embeddings.new_tensor(positive_anchors.numel()),
            "mined_negative_pairs": embeddings.new_tensor(negative_anchors.numel()),
            "active_anchors": active.sum().detach(),
            "active_anchor_fraction": active.float().sum() / denominator,
            "both_pair_anchor_fraction": (active & positive_mask & negative_mask).float().sum() / denominator,
            "mined_anchor_fraction": (positive_mask | negative_mask).float().sum() / denominator,
        }
    return loss, components
