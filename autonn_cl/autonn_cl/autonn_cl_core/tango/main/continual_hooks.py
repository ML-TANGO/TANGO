"""Continual-learning hook implementations for advanced strategies."""
from __future__ import annotations

import hashlib
import logging
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tango.common.models.experimental import attempt_load
from tango.utils.general import colorstr

from .continual import ContinualHook

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Store a single replay sample and its metadata."""

    key: str
    image_path: Path
    label_path: Optional[Path]
    score: float
    classes: Tuple[int, ...]
    source_step: int


class ConfidenceWeightedLwFOCDMHook(ContinualHook):
    """Combine confidence-weighted LwF and OCDM-style memory management."""

    METHOD_SUMMARY = 'LwF(conf-weighted) + OCDM replay'

    def __init__(
        self,
        temperature: float = 2.0,
        lambda_distill: float = 0.5,
        max_per_class: int = 24,
        max_global_samples: int = 512,
        min_confidence: float = 1e-3,
    ) -> None:
        self.temperature = float(max(temperature, 1e-3))
        self.lambda_distill = float(max(lambda_distill, 0.0))
        self.max_per_class = int(max_per_class)
        self.max_global_samples = int(max_global_samples)
        self.min_confidence = float(min_confidence)

        self.teacher: Optional[nn.Module] = None
        self.teacher_device: Optional[torch.device] = None
        self.task_type: str = ""

        self.memory_root: Optional[Path] = None
        self.memory_images_dir: Optional[Path] = None
        self.memory_labels_dir: Optional[Path] = None
        self.memory_entries: Dict[int, List[MemoryEntry]] = defaultdict(list)
        self.memory_lookup: Dict[str, MemoryEntry] = {}

        self._current_dataset = None
        self._label_lookup: Dict[Path, Path] = {}
        self._step_index: Optional[int] = None
        self._step_slug: str = ""
        self._device: Optional[torch.device] = None

        self._cached_teacher_logits: Optional[List[torch.Tensor]] = None
        self._cached_teacher_confidence: Optional[torch.Tensor] = None
        self._last_batch_paths: Sequence[str] = ()
        self._memory_candidates: List[Tuple[str, float]] = []
        self._last_distill_loss: float = 0.0

    # ------------------------------------------------------------------ helpers
    def _ensure_dirs(self) -> None:
        if self.memory_root is None:
            raise RuntimeError("Memory root not initialised")
        for directory in (self.memory_root, self.memory_images_dir, self.memory_labels_dir):
            if directory and not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)

    def _reset_step_buffers(self) -> None:
        self._cached_teacher_logits = None
        self._cached_teacher_confidence = None
        self._memory_candidates = []
        self._last_batch_paths = ()
        self._last_distill_loss = 0.0

    def _load_teacher(self, weights_path: Optional[str]) -> None:
        if not weights_path:
            self.teacher = None
            return
        path = Path(weights_path)
        if not path.exists():
            logger.warning("Continual hook: teacher checkpoint %s does not exist", path)
            self.teacher = None
            return
        try:
            teacher = attempt_load(str(path), map_location='cpu', fused=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Continual hook: failed to load teacher from %s (%s)", path, exc)
            self.teacher = None
            return

        teacher.float()
        teacher.requires_grad_(False)
        teacher.train()  # ensure Detect/Segmentation head stays in training branch
        for module in teacher.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
        self.teacher = teacher
        logger.info("Continual hook: loaded teacher checkpoint %s", path)

        if self._device is not None:
            self.teacher.to(self._device)
            self.teacher_device = self._device
        else:
            self.teacher_device = torch.device('cpu')

    def _teacher_ready(self) -> bool:
        return self.teacher is not None and self.lambda_distill > 0.0

    @staticmethod
    def _flatten_detection_cls(preds: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        cls_logits: List[torch.Tensor] = []
        for layer in preds:
            if layer.ndim < 5:
                continue
            cls_logits.append(layer[..., 5:])
        return cls_logits

    @staticmethod
    def _flatten_segmentation_cls(cls_outputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return list(cls_outputs)

    def _extract_cls_logits(self, preds: Sequence[torch.Tensor] | Tuple) -> Optional[List[torch.Tensor]]:
        # YOLO detection returns list of tensors; segmentation returns tuple of lists
        if isinstance(preds, (list, tuple)):
            if len(preds) == 4 and isinstance(preds[1], (list, tuple)):
                return self._flatten_segmentation_cls(preds[1])
            return self._flatten_detection_cls(preds)
        return None

    def _compute_confidence(self, cls_logits: List[torch.Tensor]) -> torch.Tensor:
        """Return per-image confidence averaged across anchors/spatial locations."""
        confidences: List[torch.Tensor] = []
        for logits in cls_logits:
            if logits.ndim == 5:  # detection head -> (bs, na, ny, nx, nc)
                probs = torch.sigmoid(logits / self.temperature)
                conf = probs.max(dim=-1).values  # (bs, na, ny, nx)
                confidences.append(conf.flatten(1).mean(dim=1))
            elif logits.ndim == 4:  # segmentation head -> (bs, nc, h, w)
                probs = torch.sigmoid(logits / self.temperature)
                conf = probs.max(dim=1).values  # (bs, h, w)
                confidences.append(conf.flatten(1).mean(dim=1))
            else:
                probs = torch.sigmoid(logits / self.temperature)
                conf = probs.max(dim=-1).values
                confidences.append(conf.flatten(1).mean(dim=1))
        if confidences:
            stacked = torch.stack(confidences, dim=0).mean(dim=0)
            return stacked
        return torch.zeros(1, device=cls_logits[0].device if cls_logits else 'cpu')

    def _distillation_loss(
        self,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        teacher_confidence: torch.Tensor,
    ) -> torch.Tensor:
        kd_losses: List[torch.Tensor] = []
        temp = self.temperature
        eps = self.min_confidence
        for s_logits, t_logits in zip(student_logits, teacher_logits):
            if s_logits.shape != t_logits.shape:
                min_shape = [min(sa, ta) for sa, ta in zip(s_logits.shape, t_logits.shape)]
                slicer = tuple(slice(0, m) for m in min_shape)
                s_view = s_logits[slicer]
                t_view = t_logits[slicer]
            else:
                s_view, t_view = s_logits, t_logits

            if s_view.ndim == 5:  # detection
                s_flat = s_view.reshape(s_view.shape[0], -1, s_view.shape[-1])
                t_flat = t_view.reshape(t_view.shape[0], -1, t_view.shape[-1])
            elif s_view.ndim == 4:  # segmentation head classification map (bs, nc, h, w)
                s_flat = s_view.flatten(2).transpose(1, 2)  # (bs, hw, nc)
                t_flat = t_view.flatten(2).transpose(1, 2)
            else:
                s_flat = s_view.reshape(s_view.shape[0], -1, s_view.shape[-1])
                t_flat = t_view.reshape(t_view.shape[0], -1, t_view.shape[-1])

            teacher_prob = torch.sigmoid(t_flat / temp)
            weight = torch.clamp(teacher_prob.max(dim=-1).values, min=eps)
            loss_raw = F.binary_cross_entropy_with_logits(
                s_flat / temp,
                teacher_prob,
                reduction='none',
            )
            loss_per_anchor = loss_raw.mean(dim=-1) * weight
            kd_losses.append(loss_per_anchor.mean())

        if kd_losses:
            return torch.stack(kd_losses).mean()
        return torch.tensor(0.0, device=teacher_confidence.device if teacher_confidence is not None else 'cpu')

    def _memory_quota_reached(self) -> bool:
        total = sum(len(entries) for entries in self.memory_entries.values())
        return total >= self.max_global_samples

    def _make_key(self, path: Path) -> str:
        return hashlib.sha1(str(path).encode('utf-8')).hexdigest()

    def _add_memory_entry(self, entry: MemoryEntry) -> None:
        # Respect global budget first
        if self._memory_quota_reached() and entry.key not in self.memory_lookup:
            # Remove lowest-score sample globally
            lowest_class = None
            lowest_idx = None
            lowest_score = math.inf
            for cls_id, entries in self.memory_entries.items():
                for idx, stored in enumerate(entries):
                    if stored.score < lowest_score:
                        lowest_class, lowest_idx, lowest_score = cls_id, idx, stored.score
            if lowest_class is not None and lowest_idx is not None:
                removed = self.memory_entries[lowest_class].pop(lowest_idx)
                self.memory_lookup.pop(removed.key, None)
                self._delete_sample_files(removed)

        target_bucket = self.memory_entries[entry.classes[0]] if entry.classes else self.memory_entries[-1]
        if entry.key in self.memory_lookup:
            existing = self.memory_lookup[entry.key]
            existing.score = max(existing.score, entry.score)
            self.memory_lookup[entry.key] = existing
            return
        if len(target_bucket) >= self.max_per_class:
            worst_idx = min(range(len(target_bucket)), key=lambda i: target_bucket[i].score)
            if target_bucket[worst_idx].score >= entry.score:
                return
            removed = target_bucket.pop(worst_idx)
            self.memory_lookup.pop(removed.key, None)
            self._delete_sample_files(removed)
        target_bucket.append(entry)
        self.memory_lookup[entry.key] = entry

    def _delete_sample_files(self, entry: MemoryEntry) -> None:
        try:
            if entry.image_path.exists():
                entry.image_path.unlink()
            if entry.label_path and entry.label_path.exists():
                entry.label_path.unlink()
        except Exception as exc:  # pragma: no cover - file system guard
            logger.debug("Continual hook: failed to delete memory file %s (%s)", entry.image_path, exc)

    def _copy_to_memory(self, image_path: Path, label_path: Optional[Path], step_index: int, score: float,
                        classes: Iterable[int]) -> Optional[MemoryEntry]:
        if self.memory_images_dir is None or self.memory_labels_dir is None:
            return None
        key = self._make_key(image_path)
        dst_image = self.memory_images_dir / f"{key}{image_path.suffix.lower()}"
        dst_label = self.memory_labels_dir / f"{key}.txt"

        try:
            shutil.copy2(image_path, dst_image)
            if label_path and label_path.exists():
                shutil.copy2(label_path, dst_label)
            elif dst_label.exists():
                dst_label.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Continual hook: failed to copy sample %s (%s)", image_path, exc)
            return None

        entry = MemoryEntry(
            key=key,
            image_path=dst_image,
            label_path=dst_label if label_path and label_path.exists() else None,
            score=score,
            classes=tuple(classes),
            source_step=step_index,
        )
        return entry

    def _label_path_for(self, image_path: str) -> Optional[Path]:
        if not self._label_lookup:
            return None
        normalized = Path(image_path)
        if normalized in self._label_lookup:
            return self._label_lookup[normalized]
        # try resolved absolute path and stem variations
        resolved = normalized.resolve() if normalized.exists() else normalized
        for candidate in (normalized, resolved):
            if candidate in self._label_lookup:
                return self._label_lookup[candidate]
        return None

    # ------------------------------------------------------------------ events
    def on_schedule_start(self, schedule, proj_info=None, base_save_dir=None, **_):
        self.task_type = str(proj_info.get('task_type', '')).lower() if proj_info else ''
        if base_save_dir is not None:
            base_path = Path(base_save_dir)
        else:
            base_path = Path('./runs/continual')
        self.memory_root = base_path / 'memory_bank'
        self.memory_images_dir = self.memory_root / 'images'
        self.memory_labels_dir = self.memory_root / 'labels'
        self._ensure_dirs()
        logger.info(colorstr('Continual: ') + 'LwF+OCDM hook initialised at %s', self.memory_root)

    def prepare_step_data(self, step=None, data=None, base_save_dir=None, **_):
        if not data:
            return data
        if not self.memory_lookup:
            return data
        train_path = data.get('train')
        memory_images = str(self.memory_images_dir) if self.memory_images_dir else None
        if not memory_images:
            return data
        combined: List[str]
        if isinstance(train_path, (list, tuple)):
            combined = list(train_path)
        else:
            combined = [train_path]
        if memory_images not in combined:
            combined.append(memory_images)
        updated = dict(data)
        updated['train'] = combined
        return updated

    def on_step_start(self, step, previous_weights=None, **context):
        self._step_index = step.index if step else None
        self._step_slug = step.name if step else ''
        self._reset_step_buffers()
        seen = context.get('seen_classes')
        logger.info(colorstr('Continual: ') + 'starting step %s (seen=%s)', step.name if step else 'n/a', seen)
        self._load_teacher(previous_weights)

    def on_train_start(self, model=None, dataset=None, dataloader=None, **_):
        if model is not None:
            self._device = next(model.parameters()).device
            if self.teacher is not None:
                self.teacher.to(self._device)
                self.teacher_device = self._device
        self._current_dataset = dataset
        self._label_lookup = {}
        if dataset is not None and hasattr(dataset, 'im_files') and hasattr(dataset, 'label_files'):
            try:
                for img_p, lbl_p in zip(dataset.im_files, dataset.label_files):
                    self._label_lookup[Path(img_p)] = Path(lbl_p)
            except Exception:  # pragma: no cover - dataset guard
                self._label_lookup = {}
        self._reset_step_buffers()

    def on_batch_start(self, batch=None, model=None, **_):
        if not self._teacher_ready() or batch is None or model is None:
            self._cached_teacher_logits = None
            self._cached_teacher_confidence = None
            self._last_batch_paths = batch[2] if isinstance(batch, (list, tuple)) and len(batch) > 2 else ()
            return
        imgs = batch[0]
        paths = batch[2] if isinstance(batch, (list, tuple)) and len(batch) > 2 else ()
        self._last_batch_paths = paths
        if not isinstance(imgs, torch.Tensor):
            self._cached_teacher_logits = None
            self._cached_teacher_confidence = None
            return
        device = self.teacher_device or self._device or imgs.device
        imgs_device = imgs.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
        if self.teacher is None:
            return
        with torch.no_grad():
            teacher_preds = self.teacher(imgs_device)
        cls_logits = self._extract_cls_logits(teacher_preds)
        if not cls_logits:
            self._cached_teacher_logits = None
            self._cached_teacher_confidence = None
            return
        self._cached_teacher_logits = [tensor.detach() for tensor in cls_logits]
        conf = self._compute_confidence(self._cached_teacher_logits)
        self._cached_teacher_confidence = conf.detach()

    def on_before_backward(self, pred=None, loss=None, loss_items=None, **_):
        if not self._teacher_ready() or self._cached_teacher_logits is None or pred is None:
            return
        student_logits = self._extract_cls_logits(pred)
        if not student_logits:
            return
        teacher_conf = self._cached_teacher_confidence
        if teacher_conf is None:
            return
        kd_loss = self._distillation_loss(student_logits, self._cached_teacher_logits, teacher_conf)
        if kd_loss is None:
            return
        scaled_loss = kd_loss * self.lambda_distill
        if isinstance(loss, torch.Tensor):
            loss += scaled_loss
            self._last_distill_loss = float(scaled_loss.detach().cpu())
        if isinstance(loss_items, torch.Tensor) and loss_items.numel() >= 3:
            cls_index = 1 if loss_items.numel() >= 4 and self.task_type == 'segmentation' else 2
            cls_index = min(cls_index, loss_items.numel() - 1)
            loss_items[cls_index] += scaled_loss.detach()

    def on_batch_end(self, **context):
        if not self._teacher_ready():
            return
        paths = context.get('paths') or self._last_batch_paths
        if not paths or self._cached_teacher_confidence is None:
            return
        conf = self._cached_teacher_confidence.detach().cpu()
        if conf.numel() != len(paths):
            return
        for path, score in zip(paths, conf.tolist()):
            priority = 1.0 - max(score, self.min_confidence)
            self._memory_candidates.append((path, priority))

    def on_step_end(self, **_):
        if not self._memory_candidates:
            return
        self._ensure_dirs()
        step_idx = self._step_index if self._step_index is not None else -1
        processed = 0
        for path, priority in sorted(self._memory_candidates, key=lambda item: item[1], reverse=True):
            image_path = Path(path)
            label_path = self._label_path_for(path)
            classes: Tuple[int, ...] = ()
            if label_path and label_path.exists():
                try:
                    with open(label_path, 'r', encoding='utf-8') as label_file:
                        class_ids = {int(float(line.split()[0])) for line in label_file if line.strip()}
                    classes = tuple(sorted(class_ids))
                except Exception:  # pragma: no cover - label parse guard
                    classes = ()
            entry = self._copy_to_memory(image_path, label_path, step_idx, priority, classes or (-1,))
            if not entry:
                continue
            target_cls = entry.classes[0] if entry.classes else -1
            self._add_memory_entry(entry)
            processed += 1
            if processed >= self.max_global_samples:
                break
        self._reset_step_buffers()

    def on_schedule_end(self, **_):
        summary = sum(len(v) for v in self.memory_entries.values())
        logger.info(colorstr('Continual: ') + 'memory bank size=%d, lambda=%.3f, T=%.2f',
                    summary, self.lambda_distill, self.temperature)

__all__ = ["ConfidenceWeightedLwFOCDMHook", "MemoryEntry"]
