import copy
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .train import train

logger = logging.getLogger(__name__)


@dataclass
class ContinualStep:
    """Container describing a single continual-learning step."""

    index: int
    name: str
    class_ids: List[int]
    class_names: List[str]
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    epochs: Optional[int] = None
    id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContinualHook:
    """Base class for continual-learning lifecycle hooks."""

    def on_schedule_start(self, schedule: Sequence[ContinualStep], **_: Any) -> None:  # noqa: D401
        """Called once before any step is executed."""

    def on_schedule_end(self, schedule: Sequence[ContinualStep], **_: Any) -> None:
        """Called after the last continual step finishes."""

    def on_step_start(self, step: ContinualStep, **_: Any) -> None:
        """Called before invoking training for an individual step."""

    def on_step_end(self, step: ContinualStep, **_: Any) -> None:
        """Called once the training for a step completes."""

    def on_train_start(self, **_: Any) -> None:
        """Forwarded directly to the underlying training loop."""

    def on_train_end(self, **_: Any) -> None:
        """Forwarded directly to the underlying training loop."""

    def on_epoch_start(self, **_: Any) -> None:
        """Forwarded event from the training loop."""

    def on_epoch_end(self, **_: Any) -> None:
        """Forwarded event from the training loop."""

    def on_batch_start(self, **_: Any) -> None:
        """Forwarded event from the training loop."""

    def on_before_backward(self, **_: Any) -> None:
        """Forwarded event from the training loop before backprop."""

    def on_batch_end(self, **_: Any) -> None:
        """Forwarded event from the training loop."""


class LoggingHook(ContinualHook):
    """Minimal hook that logs high-level continual-learning milestones."""

    def on_schedule_start(self, schedule: Sequence[ContinualStep], **_: Any) -> None:
        logger.info("Continual schedule starting with %d steps", len(schedule))

    def on_schedule_end(self, schedule: Sequence[ContinualStep], **context: Any) -> None:
        final_model = context.get('final_model')
        logger.info("Continual schedule finished. Final model: %s", final_model)

    def on_step_start(self, step: ContinualStep, **context: Any) -> None:
        seen = context.get('seen_classes')
        logger.info(
            "Continual step %s (index=%d) starting with classes=%s",
            step.name,
            step.index,
            step.class_ids,
        )
        if seen:
            logger.info("Previously seen classes: %s", seen)

    def on_step_end(self, step: ContinualStep, **context: Any) -> None:
        results = context.get('results')
        logger.info("Continual step %s finished. Last results=%s", step.name, results)


class ContinualHookManager:
    """Composite hook dispatcher used by the training orchestrator."""

    def __init__(self, hooks: Optional[Iterable[ContinualHook]] = None):
        self._hooks: List[ContinualHook] = list(hooks) if hooks else []

    def add_hook(self, hook: ContinualHook) -> None:
        self._hooks.append(hook)

    def dispatch(self, event: str, **kwargs: Any) -> Optional[Any]:
        result: Optional[Any] = None
        for hook in self._hooks:
            handler = getattr(hook, event, None)
            if callable(handler):
                result = handler(**kwargs)
        return result

    def __len__(self) -> int:
        return len(self._hooks)


def _slugify(value: str) -> str:
    normalized = re.sub(r'[^a-zA-Z0-9]+', '-', value.strip().lower())
    return normalized.strip('-') or 'step'


def _ensure_manager(hooks: Optional[Union[ContinualHook, Iterable[ContinualHook], ContinualHookManager]]) -> ContinualHookManager:
    if hooks is None:
        return ContinualHookManager([LoggingHook()])
    if isinstance(hooks, ContinualHookManager):
        return hooks
    if isinstance(hooks, ContinualHook):
        return ContinualHookManager([hooks])
    return ContinualHookManager(list(hooks))


def parse_continual_schedule(data_dict: Dict[str, Any]) -> List[ContinualStep]:
    cfg = data_dict.get('continual') or {}
    raw_steps = cfg.get('steps', [])
    if not raw_steps:
        return []

    class_names: Sequence[str] = data_dict.get('names', [])
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    default_epochs = cfg.get('default_epochs')

    parsed_steps: List[ContinualStep] = []
    for idx, step_cfg in enumerate(raw_steps):
        provided_ids = step_cfg.get('class_ids')
        provided_names = step_cfg.get('class_names') or []

        if provided_ids is None and not provided_names:
            raise ValueError(f"Continual step #{idx} requires 'class_ids' or 'class_names'")

        if provided_ids is None:
            missing = [name for name in provided_names if name not in name_to_id]
            if missing:
                raise ValueError(f"Unknown class names in continual step #{idx}: {missing}")
            class_ids = [name_to_id[name] for name in provided_names]
        else:
            class_ids = [int(cid) for cid in provided_ids]
            provided_names = [class_names[cid] for cid in class_ids]

        metadata = {
            key: value
            for key, value in step_cfg.items()
            if key not in {'id', 'name', 'class_ids', 'class_names', 'train', 'val', 'epochs'}
        }

        parsed_steps.append(
            ContinualStep(
                index=idx,
                name=step_cfg.get('name', f'step_{idx:02d}'),
                id=step_cfg.get('id', idx),
                class_ids=class_ids,
                class_names=provided_names,
                train_path=step_cfg.get('train'),
                val_path=step_cfg.get('val'),
                epochs=step_cfg.get('epochs', default_epochs),
                metadata=metadata,
            )
        )

    return parsed_steps


def train_continual(
    proj_info: Dict[str, Any],
    hyp: Dict[str, Any],
    opt: Any,
    data_dict: Dict[str, Any],
    tb_writer: Any = None,
    hooks: Optional[Union[ContinualHook, Iterable[ContinualHook], ContinualHookManager]] = None,
):
    """Run continual-learning schedule, falling back to single-step training if needed."""

    schedule = parse_continual_schedule(data_dict)
    if not schedule:
        return train(proj_info, hyp, opt, data_dict, tb_writer)

    hook_manager = _ensure_manager(hooks)
    hook_manager.dispatch('on_schedule_start', schedule=schedule, proj_info=proj_info)

    base_save_dir = Path(opt.save_dir)
    base_save_dir.mkdir(parents=True, exist_ok=True)

    seen_class_ids: List[int] = []
    previous_weights = getattr(opt, 'weights', None)
    final_model: Optional[Path] = None
    results: Optional[Any] = None

    for step in schedule:
        step_index = step.index
        step_slug = _slugify(step.name)
        step_opt = copy.deepcopy(opt)
        step_opt.save_dir = str(base_save_dir / f'step_{step_index:02d}_{step_slug}')
        step_opt.resume = bool(getattr(opt, 'resume', False)) if step_index == 0 else False
        step_opt.weights = previous_weights or getattr(opt, 'weights', None)
        step_opt.epochs = step.epochs or getattr(opt, 'epochs', None)
        if not hasattr(step_opt, 'bs_factor'):
            step_opt.bs_factor = getattr(opt, 'bs_factor', 0.8)

        step_data = copy.deepcopy(data_dict)
        step_data['train'] = step.train_path or data_dict.get('train')
        step_data['val'] = step.val_path or data_dict.get('val')
        step_data['class_filter'] = step.class_ids

        seen_class_ids = sorted(set(seen_class_ids).union(step.class_ids))
        step_data['eval_class_filter'] = seen_class_ids

        step_meta = {
            'index': step.index,
            'id': step.id,
            'name': step.name,
            'epochs': step.epochs,
            'class_ids': step.class_ids,
            'class_names': step.class_names,
            'seen_class_ids': seen_class_ids,
            'total_steps': len(schedule),
            'metadata': step.metadata,
        }
        step_data['continual_step'] = step_meta

        hook_manager.dispatch(
            'on_step_start',
            step=step,
            seen_classes=seen_class_ids,
            previous_weights=previous_weights,
        )

        results, final_model = train(
            proj_info,
            hyp,
            step_opt,
            step_data,
            tb_writer,
            continual_hooks=hook_manager,
        )

        hook_manager.dispatch(
            'on_step_end',
            step=step,
            results=results,
            model_path=final_model,
            seen_classes=seen_class_ids,
        )

        last_checkpoint = Path(final_model).with_name('last.pt') if final_model else None
        if last_checkpoint and last_checkpoint.exists():
            previous_weights = str(last_checkpoint)
        elif final_model:
            previous_weights = str(final_model)
        else:
            previous_weights = getattr(opt, 'weights', None)

        opt.save_dir = step_opt.save_dir
        opt.weights = previous_weights
        opt.bs_factor = getattr(step_opt, 'bs_factor', getattr(opt, 'bs_factor', 0.8))

    hook_manager.dispatch(
        'on_schedule_end',
        schedule=schedule,
        final_model=str(final_model) if final_model else None,
        results=results,
        seen_classes=seen_class_ids,
    )

    return results, final_model
