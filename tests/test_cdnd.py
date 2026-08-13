from __future__ import annotations

import torch

from a2b_lightvpr.data import PKBalancedBatchSampler
from a2b_lightvpr.vpr.losses import cross_domain_neighborhood_loss
from a2b_lightvpr.vpr.neighborhood import DatasetNeighborhoodQueue


def _rows():
    rows = []
    for dataset in ("a", "b"):
        for place in range(4):
            for image in range(3):
                rows.append(
                    {
                        "dataset": dataset,
                        "global_place_id": f"{dataset}:p{place}",
                        "image_id": f"{dataset}-{place}-{image}",
                    }
                )
    return rows


def test_pk_sampler_builds_single_dataset_four_by_two_batches_and_restores_cursor():
    rows = _rows()
    sampler = PKBalancedBatchSampler(rows, {"a": 16, "b": 16}, 4, 2)
    iterator = iter(sampler)
    first = next(iterator)
    state = sampler.state_dict()
    remaining = list(iterator)

    for batch in [first, *remaining]:
        datasets = {rows[index]["dataset"] for index in batch}
        places = [rows[index]["global_place_id"] for index in batch]
        assert len(batch) == 8
        assert len(datasets) == 1
        assert len(set(places)) == 4
        assert all(places.count(place) == 2 for place in set(places))
        assert len(batch) == len(set(batch))

    resumed = PKBalancedBatchSampler(rows, {"a": 16, "b": 16}, 4, 2)
    resumed.load_state_dict(state)
    assert list(resumed) == remaining

    next_epoch = PKBalancedBatchSampler(rows, {"a": 16, "b": 16}, 4, 2)
    next_epoch.set_epoch(1)
    assert [first, *remaining] != list(next_epoch)


def test_dataset_queue_is_fifo_isolated_and_round_trips():
    queue = DatasetNeighborhoodQueue(("a", "b"), capacity=3, device=torch.device("cpu"))
    for offset in (0, 2):
        teacher = torch.arange(offset, offset + 4, dtype=torch.float32).reshape(2, 2)
        student = teacher + 10
        labels = torch.tensor([offset, offset + 1])
        keys = torch.tensor([offset + 100, offset + 101])
        queue.enqueue("a", teacher, student, labels, keys)
    assert queue.size("a") == 3
    assert queue.size("b") == 0

    current = torch.tensor([[99.0, 98.0]])
    candidates = queue.candidates("a", current, current, torch.tensor([9]), torch.tensor([999]))
    assert candidates.labels.tolist() == [9, 1, 2, 3]
    assert candidates.image_keys.tolist() == [999, 101, 102, 103]
    only_b = queue.candidates("b", current, current, torch.tensor([9]), torch.tensor([999]))
    assert only_b.labels.tolist() == [9]

    restored = DatasetNeighborhoodQueue(("a", "b"), capacity=3, device=torch.device("cpu"))
    restored.load_state_dict(queue.state_dict())
    assert restored.size("a") == 3
    assert restored.candidates(
        "a", current, current, torch.tensor([9]), torch.tensor([999])
    ).image_keys.tolist() == candidates.image_keys.tolist()


def test_cdnd_masks_self_and_backpropagates_only_into_student_descriptors():
    teacher = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    fake = torch.tensor(
        [[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]],
        requires_grad=True,
    )
    day = torch.tensor(
        [[0.75, 0.25], [0.65, 0.35], [0.25, 0.75], [0.35, 0.65]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    keys = torch.arange(4)
    fake_loss, day_loss, diagnostics = cross_domain_neighborhood_loss(
        teacher,
        teacher,
        fake,
        day,
        day,
        labels,
        labels,
        keys,
        keys,
    )
    assert torch.isfinite(fake_loss + day_loss)
    assert diagnostics["valid_anchors"].item() == 4
    (fake_loss + 0.5 * day_loss).backward()
    assert fake.grad is not None and fake.grad.abs().sum() > 0
    assert day.grad is not None and day.grad.abs().sum() > 0
    assert teacher.grad is None


def test_cdnd_returns_differentiable_zero_without_valid_anchors():
    teacher = torch.randn(2, 3)
    fake = torch.randn(2, 3, requires_grad=True)
    day = torch.randn(2, 3, requires_grad=True)
    labels = torch.zeros(2, dtype=torch.long)
    keys = torch.arange(2)
    fake_loss, day_loss, diagnostics = cross_domain_neighborhood_loss(
        teacher, teacher, fake, day, day, labels, labels, keys, keys
    )
    assert fake_loss.item() == 0.0
    assert day_loss.item() == 0.0
    assert diagnostics["valid_anchors"].item() == 0
    (fake_loss + day_loss).backward()
    assert fake.grad is not None
    assert day.grad is not None


def test_sampler_and_queue_restore_produce_identical_next_relation_loss():
    rows = _rows()
    sampler = PKBalancedBatchSampler(rows, {"a": 16, "b": 16}, 4, 2)
    iterator = iter(sampler)
    first = next(iterator)
    queue = DatasetNeighborhoodQueue(("a", "b"), capacity=16, device=torch.device("cpu"))

    def tensors(indices):
        values = torch.tensor(indices, dtype=torch.float32)
        teacher = torch.stack((torch.sin(values), torch.cos(values)), dim=1)
        student = torch.stack((torch.sin(values + 0.2), torch.cos(values + 0.2)), dim=1)
        labels = torch.tensor(
            [int(rows[index]["global_place_id"].rsplit("p", 1)[1]) for index in indices]
        )
        keys = torch.tensor(indices)
        return teacher, student, labels, keys

    queue.enqueue("a" if rows[first[0]]["dataset"] == "a" else "b", *tensors(first))
    sampler_state = sampler.state_dict()
    queue_state = queue.state_dict()
    next_batch = next(iterator)

    restored_sampler = PKBalancedBatchSampler(rows, {"a": 16, "b": 16}, 4, 2)
    restored_sampler.load_state_dict(sampler_state)
    restored_queue = DatasetNeighborhoodQueue(("a", "b"), 16, torch.device("cpu"))
    restored_queue.load_state_dict(queue_state)
    assert next(iter(restored_sampler)) == next_batch

    dataset = rows[next_batch[0]]["dataset"]
    teacher, student, labels, keys = tensors(next_batch)
    first_candidates = queue.candidates(dataset, teacher, student, labels, keys)
    restored_candidates = restored_queue.candidates(dataset, teacher, student, labels, keys)
    first_loss = cross_domain_neighborhood_loss(
        teacher, first_candidates.teacher, student, student,
        first_candidates.student_day, labels, first_candidates.labels,
        keys, first_candidates.image_keys,
    )[:2]
    restored_loss = cross_domain_neighborhood_loss(
        teacher, restored_candidates.teacher, student, student,
        restored_candidates.student_day, labels, restored_candidates.labels,
        keys, restored_candidates.image_keys,
    )[:2]
    assert torch.equal(torch.stack(first_loss), torch.stack(restored_loss))
