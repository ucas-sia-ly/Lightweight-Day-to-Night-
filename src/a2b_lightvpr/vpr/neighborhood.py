from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NeighborhoodCandidates:
    teacher: torch.Tensor
    student_day: torch.Tensor
    labels: torch.Tensor
    image_keys: torch.Tensor


class DatasetNeighborhoodQueue:
    """Per-dataset FIFO memory used only by training-time relation distillation."""

    def __init__(self, datasets: tuple[str, ...], capacity: int, device: torch.device):
        self.datasets = tuple(sorted(str(name) for name in datasets))
        self.capacity = int(capacity)
        self.device = torch.device(device)
        if self.capacity < 0:
            raise ValueError("Neighborhood queue capacity cannot be negative")
        self._entries: dict[str, dict[str, torch.Tensor]] = {
            name: {} for name in self.datasets
        }

    def __len__(self) -> int:
        return sum(self.size(name) for name in self.datasets)

    def size(self, dataset: str) -> int:
        self._validate_dataset(dataset)
        labels = self._entries[dataset].get("labels")
        return 0 if labels is None else int(labels.shape[0])

    def _validate_dataset(self, dataset: str) -> None:
        if dataset not in self._entries:
            raise KeyError(f"Unknown neighborhood queue dataset: {dataset}")

    def candidates(
        self,
        dataset: str,
        teacher: torch.Tensor,
        student_day: torch.Tensor,
        labels: torch.Tensor,
        image_keys: torch.Tensor,
    ) -> NeighborhoodCandidates:
        self._validate_dataset(dataset)
        entry = self._entries[dataset]
        if not entry:
            return NeighborhoodCandidates(teacher, student_day, labels, image_keys)
        if entry["teacher"].shape[1] != teacher.shape[1]:
            raise ValueError("Teacher descriptor dimension changed within a queue")
        if entry["student_day"].shape[1] != student_day.shape[1]:
            raise ValueError("Student descriptor dimension changed within a queue")
        return NeighborhoodCandidates(
            teacher=torch.cat((teacher, entry["teacher"].to(dtype=teacher.dtype)), dim=0),
            student_day=torch.cat(
                (student_day, entry["student_day"].to(dtype=student_day.dtype)), dim=0
            ),
            labels=torch.cat((labels, entry["labels"]), dim=0),
            image_keys=torch.cat((image_keys, entry["image_keys"]), dim=0),
        )

    @torch.no_grad()
    def enqueue(
        self,
        dataset: str,
        teacher: torch.Tensor,
        student_day: torch.Tensor,
        labels: torch.Tensor,
        image_keys: torch.Tensor,
    ) -> None:
        self._validate_dataset(dataset)
        if self.capacity == 0:
            return
        values = {
            "teacher": teacher.detach().to(self.device, dtype=torch.float16),
            "student_day": student_day.detach().to(self.device),
            "labels": labels.detach().to(self.device, dtype=torch.long),
            "image_keys": image_keys.detach().to(self.device, dtype=torch.long),
        }
        batch = int(values["labels"].shape[0])
        if any(int(value.shape[0]) != batch for value in values.values()):
            raise ValueError("Neighborhood queue fields must have the same leading dimension")
        entry = self._entries[dataset]
        if entry:
            if entry["teacher"].shape[1] != values["teacher"].shape[1]:
                raise ValueError("Teacher descriptor dimension changed within a queue")
            if entry["student_day"].shape[1] != values["student_day"].shape[1]:
                raise ValueError("Student descriptor dimension changed within a queue")
            values = {
                key: torch.cat((entry[key], value), dim=0)[-self.capacity :]
                for key, value in values.items()
            }
        else:
            values = {key: value[-self.capacity :] for key, value in values.items()}
        self._entries[dataset] = values

    def state_dict(self) -> dict:
        return {
            "kind": "dataset_neighborhood_queue",
            "capacity": self.capacity,
            "datasets": self.datasets,
            "entries": {
                dataset: {key: value.detach().cpu() for key, value in entry.items()}
                for dataset, entry in self._entries.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != "dataset_neighborhood_queue":
            raise ValueError("Unsupported neighborhood queue state")
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("Neighborhood queue capacity does not match the checkpoint")
        if tuple(state.get("datasets", ())) != self.datasets:
            raise ValueError("Neighborhood queue datasets do not match the checkpoint")
        entries = state.get("entries", {})
        if set(entries) != set(self.datasets):
            raise ValueError("Neighborhood queue checkpoint is missing a dataset")
        restored: dict[str, dict[str, torch.Tensor]] = {}
        for dataset in self.datasets:
            entry = entries[dataset]
            if not entry:
                restored[dataset] = {}
                continue
            if set(entry) != {"teacher", "student_day", "labels", "image_keys"}:
                raise ValueError("Neighborhood queue checkpoint has invalid fields")
            size = int(entry["labels"].shape[0])
            if size > self.capacity or any(int(value.shape[0]) != size for value in entry.values()):
                raise ValueError("Neighborhood queue checkpoint has invalid tensor sizes")
            restored[dataset] = {
                "teacher": entry["teacher"].to(self.device, dtype=torch.float16),
                "student_day": entry["student_day"].to(self.device),
                "labels": entry["labels"].to(self.device, dtype=torch.long),
                "image_keys": entry["image_keys"].to(self.device, dtype=torch.long),
            }
        self._entries = restored
