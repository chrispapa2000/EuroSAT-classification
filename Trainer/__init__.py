"""
trainer.py
----------
Abstract base Trainer for PyTorch projects.

Subclass BaseTrainer and implement the six abstract methods:
  - build_datasets()
  - build_model()
  - compute_loss()
  - forward_step()
  - compute_metrics()          (optional override)
  - on_epoch_end()             (optional hook)

Everything else — optimiser / scheduler creation, the train / val loops,
gradient scaling, checkpointing, TensorBoard logging and resume logic — is
handled here and stays out of your subclass.
"""

from __future__ import annotations

import abc
import logging
import os
import time
from pathlib import Path
from typing import Any
import importlib

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf, DictConfig
from torch.amp import GradScaler
from torch.amp import autocast
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file and return an OmegaConf DictConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return OmegaConf.create(raw)


class AverageMeter:
    """Tracks a running mean of a scalar (loss, accuracy, …)."""

    def __init__(self, name: str = "meter"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------

class BaseTrainer(abc.ABC):
    """
    Abstract trainer that wires together:
      data  →  model  →  optimizer / scheduler  →  train loop
           →  val loop  →  checkpointing  →  TensorBoard logging

    Parameters
    ----------
    config : str | Path | DictConfig
        Path to a YAML file or an already-parsed OmegaConf DictConfig.
    """

    def __init__(self, config: Path):
        # ── Config ────────────────────────────────────────────────────────
        if isinstance(config, (str, Path)):
            self.cfg = load_config(config)
        else:
            self.cfg = config


        self._set_seed(self.cfg.experiment.seed)

        # ── Device ────────────────────────────────────────────────────────
        self.device = self._resolve_device()

        # ── Logging / artefact dirs ────────────────────────────────────────
        self.prepare_dirs()
        
        # ── Save config inside the experiment directory ────────────────────
        config_save_path = self.experiment_dir / 'config.yml'
        with open(config_save_path, 'w+') as f: 
            OmegaConf.save(self.cfg, f)

        self._writer: SummaryWriter | None = None   # lazy-init in train()

        # ── State (populated by setup()) ──────────────────────────────────
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None
        self.scaler: GradScaler | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None

        self.start_epoch: int = 1
        self.best_metric: float = float("-inf") if self._monitor_mode == "max" else float("inf")
        self.epochs_without_improvement: int = 0
        self.global_step: int = 0
    
    def prepare_dirs(self):
        # create base directory for experiments
        self.base_dir = Path(self.cfg.experiment.basedir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # create project directory
        self.project_dir = Path(os.path.join(self.base_dir, self.cfg.experiment.project))
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # create experiment directory
        self.experiment_dir = Path(os.path.join(self.project_dir, self.cfg.experiment.name))
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # logging and checkpoint dirs        
        self.log_dir = Path(os.path.join(self.experiment_dir, self.cfg.logging.get("save_dir", "logs")))
        self.ckpt_dir = Path(os.path.join(self.experiment_dir, self.cfg.checkpointing.get("dir", "checkpoint_dir")))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    def count_params(self, model):
        n_params = sum([param.numel() for param in model.parameters()])
        logger.info(f"─── Number of model parameters: {n_params/1e+6:.3f}M ───")

    # ──────────────────────────────────────────────────────────────────────
    # Abstract interface — subclasses MUST implement these
    # ──────────────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def build_datasets(self) -> tuple[Dataset, Dataset]:
        """
        Build and return (train_dataset, val_dataset).

        Use self.cfg.data for all dataset-level knobs (root_dir, splits,
        augmentation, normalisation, …).
        """

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """
        Build, configure, and return the model.

        Use self.cfg.model for architecture choices, pretrained weights,
        frozen layers, head dimensions, etc.
        The model should NOT be moved to a device here; the base class
        handles device placement.
        """
        

    @abc.abstractmethod
    def compute_loss(
        self,
        outputs: Any,
        targets: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        """
        Given raw model outputs, ground-truth targets, and the original
        batch, return a scalar loss tensor.

        Parameters
        ----------
        outputs : Any
            Whatever your model returns (logits, dict, tuple, …).
        targets : torch.Tensor
            Ground-truth labels / masks / boxes, etc.
        batch : Any
            The full batch as returned by the DataLoader, in case you
            need auxiliary information (e.g. sample weights).
        """

    @abc.abstractmethod
    def forward_step(
        self,
        batch: Any,
    ) -> tuple[Any, torch.Tensor]:
        """
        Run a single forward pass.

        Move inputs to self.device, call self.model, and return
        (outputs, targets) so that BaseTrainer can call compute_loss.

        Example
        -------
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.model(images)
        return outputs, labels
        """

    # ──────────────────────────────────────────────────────────────────────
    # Optional hooks — override for custom behaviour
    # ──────────────────────────────────────────────────────────────────────

    def compute_metrics(
        self,
        outputs: Any,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """
        Return a dict of metric-name → float for logging.

        Default implementation returns an empty dict.
        Override to add task-specific metrics (accuracy, mAP, dice, …).
        """
        return {}

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the very beginning of each epoch (train + val)."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """
        Called after the validation loop of each epoch.

        Override to add custom epoch-level logic (e.g. scheduler warm-up
        tricks, SWA updates, EMA updates, …).
        """

    def on_train_end(self) -> None:
        """Called once after the training loop exits (incl. early stopping)."""

    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Build all components. Called automatically by train()."""
        logger.info("─── Setting up datasets ───")
        train_ds, val_ds = self.build_datasets()
        self.train_loader, self.val_loader = self._build_loaders(train_ds, val_ds)

        logger.info("─── Building model ───")
        self.model = self.build_model().to(self.device)
        if len(self.cfg.hardware.get("gpu_ids", [0])) > 1:
            self.model = nn.DataParallel(
                self.model, device_ids=self.cfg.hardware.gpu_ids
            )

        logger.info("─── Building optimiser & scheduler ───")
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        use_amp = self.cfg.training.get("precision", "fp32") in ("fp16", "bf16")
        self.scaler = GradScaler(enabled=use_amp and self.device.type == "cuda")

    # ──────────────────────────────────────────────────────────────────────
    # Main train entry-point
    # ──────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Full training loop: setup → [optional resume] → epoch loop."""
        self.setup()

        # Resume from checkpoint if requested
        resume_path = self.cfg.checkpointing.get("resume_from", None)
        if resume_path:
            self._load_checkpoint(resume_path)

        # TensorBoard writer
        tb_cfg = self.cfg.logging.get("tensorboard", {})
        self._writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=tb_cfg.get("flush_secs", 30),
        )

        total_epochs = self.cfg.training.epochs
        logger.info(
            f"Starting training: epochs {self.start_epoch}–{total_epochs}  "
            f"device={self.device}  amp={self.scaler.is_enabled()}"
        )

        for epoch in range(self.start_epoch, total_epochs + 1):
            self.on_epoch_start(epoch)

            # ── Train ──
            train_metrics = self._run_epoch(epoch, phase="train")
            self._log_metrics(train_metrics, epoch, prefix="train")

            # ── Validate ──
            eval_cfg = self.cfg.evaluation
            if epoch % eval_cfg.get("eval_every_n_epochs", 1) == 0:
                val_metrics = self._run_epoch(epoch, phase="val")
                self._log_metrics(val_metrics, epoch, prefix="val")
            else:
                val_metrics = {}

            # ── Scheduler step ──
            self._scheduler_step(val_metrics)

            # ── Log LR ──
            current_lr = self.optimizer.param_groups[0]["lr"]
            self._writer.add_scalar("train/lr", current_lr, epoch)

            # ── Hook ──
            all_metrics = {**train_metrics, **val_metrics}
            self.on_epoch_end(epoch, all_metrics)

            # ── Checkpointing ──
            monitor_key = self._monitor_key
            if monitor_key in all_metrics:
                self._maybe_save_checkpoint(epoch, all_metrics[monitor_key], all_metrics)

            # ── Early stopping ──
            if self._should_stop_early():
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        self._writer.close()
        logger.info("Training complete.")
        self.on_train_end()

    # ──────────────────────────────────────────────────────────────────────
    # Train / val epoch
    # ──────────────────────────────────────────────────────────────────────

    def _run_epoch(self, epoch: int, phase: str) -> dict[str, float]:
        is_train = phase == "train"
        self.model.train() if is_train else self.model.eval()

        loader = self.train_loader if is_train else self.val_loader
        loss_meter = AverageMeter("loss")
        metric_accumulators: dict[str, AverageMeter] = {}
        t0 = time.time()

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        precision = self.cfg.training.get("precision", "fp32")
        amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        with ctx:
            for batch_idx, batch in enumerate(loader):
                with autocast(
                    device_type=self.device.type,
                    dtype=amp_dtype,
                    enabled=self.scaler.is_enabled(),
                ):
                    outputs, targets = self.forward_step(batch)
                    loss = self.compute_loss(outputs, targets, batch)

                n = targets.size(0) if isinstance(targets, torch.Tensor) else 1
                loss_meter.update(loss.item(), n)

                if is_train:
                    accum = self.cfg.training.get("accumulate_grad_batches", 1)
                    loss = loss / accum

                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % accum == 0:
                        self.scaler.unscale_(self.optimizer)
                        clip = self.cfg.training.get("grad_clip_norm", None)
                        if clip:
                            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1

                        # Step-level TensorBoard
                        log_every = self.cfg.logging.get("log_every_n_steps", 50)
                        if self.global_step % log_every == 0 and self._writer:
                            self._writer.add_scalar(
                                "train_step/loss", loss_meter.val, self.global_step
                            )

                # Metrics
                with torch.no_grad():
                    batch_metrics = self.compute_metrics(outputs, targets)
                for k, v in batch_metrics.items():
                    if k not in metric_accumulators:
                        metric_accumulators[k] = AverageMeter(k)
                    metric_accumulators[k].update(v, n)

        elapsed = time.time() - t0
        result = {"loss": loss_meter.avg}
        result.update({k: m.avg for k, m in metric_accumulators.items()})

        logger.info(
            f"Epoch {epoch:03d} [{phase:>5s}]  "
            + "  ".join(f"{k}={v:.4f}" for k, v in result.items())
            + f"  ({elapsed:.1f}s)"
        )
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Optimiser
    # ──────────────────────────────────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.cfg.optimizer
        name = opt_cfg.name.lower()
        params = self._get_param_groups()

        if name == "sgd":
            return SGD(
                params,
                lr=opt_cfg.lr,
                momentum=opt_cfg.get("momentum", 0.9),
                weight_decay=opt_cfg.get("weight_decay", 1e-4),
                nesterov=opt_cfg.get("nesterov", False),
            )
        elif name == "adam":
            return Adam(
                params,
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
                eps=opt_cfg.get("eps", 1e-8),
                weight_decay=opt_cfg.get("weight_decay", 0.0),
            )
        elif name == "adamw":
            return AdamW(
                params,
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
                eps=opt_cfg.get("eps", 1e-8),
                weight_decay=opt_cfg.get("weight_decay", 0.05),
            )
        else:
            raise ValueError(f"Unknown optimiser: '{name}'")

    def _get_param_groups(self) -> list[dict]:
        """
        Build param groups, respecting layer-wise LR decay if configured.

        Override this method for fine-grained control (e.g. frozen layers,
        different LRs per module).
        """
        decay = self.cfg.optimizer.get("layer_lr_decay", None)
        if decay is None:
            return [{"params": self.model.parameters()}]

        # Simple layer-wise LR decay: deeper layers get smaller LR
        named_params = list(self.model.named_parameters())
        num_layers = len(named_params)
        groups = []
        for i, (name, param) in enumerate(named_params):
            if not param.requires_grad:
                continue
            layer_lr = self.cfg.optimizer.lr * (decay ** (num_layers - i))
            groups.append({"params": [param], "lr": layer_lr, "name": name})
        return groups

    # ──────────────────────────────────────────────────────────────────────
    # Scheduler
    # ──────────────────────────────────────────────────────────────────────

    def _build_scheduler(self):
        sch_cfg = self.cfg.scheduler
        name = sch_cfg.name.lower()
        opt = self.optimizer
        epochs = self.cfg.training.epochs

        if name == "cosine":
            c = sch_cfg.get("cosine", {})
            base = CosineAnnealingLR(
                opt,
                T_max=c.get("t_max", epochs),
                eta_min=c.get("eta_min", 1e-6),
            )
        elif name == "step":
            s = sch_cfg.get("step", {})
            base = StepLR(opt, step_size=s.get("step_size", 30), gamma=s.get("gamma", 0.1))
        elif name == "multistep":
            m = sch_cfg.get("multistep", {})
            base = MultiStepLR(
                opt, milestones=list(m.get("milestones", [30, 60, 80])), gamma=m.get("gamma", 0.1)
            )
        elif name == "plateau":
            p = sch_cfg.get("plateau", {})
            return ReduceLROnPlateau(
                opt,
                mode=self._monitor_mode,
                factor=p.get("factor", 0.1),
                patience=p.get("patience", 5),
                min_lr=p.get("min_lr", 1e-7),
            )
        elif name == "onecycle":
            steps_per_epoch = len(self.train_loader)
            return OneCycleLR(
                opt,
                max_lr=self.cfg.optimizer.lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
            )
        else:
            raise ValueError(f"Unknown scheduler: '{name}'")

        # Wrap with linear warmup if requested
        warmup = sch_cfg.get("warmup_epochs", 0)
        if warmup > 0:
            return _WarmupScheduler(opt, base_scheduler=base, warmup_epochs=warmup)
        return base

    def _scheduler_step(self, val_metrics: dict[str, float]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            key = self.cfg.scheduler.get("plateau", {}).get("monitor", "loss")
            metric = val_metrics.get(key, None)
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()

    # ──────────────────────────────────────────────────────────────────────
    # DataLoaders
    # ──────────────────────────────────────────────────────────────────────

    def _build_loaders(
        self, train_ds: Dataset, val_ds: Dataset
    ) -> tuple[DataLoader, DataLoader]:
        dl_cfg = self.cfg.data.loader
        common = dict(
            batch_size=dl_cfg.batch_size,
            num_workers=dl_cfg.get("num_workers", 4),
            pin_memory=dl_cfg.get("pin_memory", True),
            prefetch_factor=dl_cfg.get("prefetch_factor", 2) if dl_cfg.get("num_workers", 4) > 0 else None,
        )
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            drop_last=dl_cfg.get("drop_last", True),
            **common,
        )
        val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
        return train_loader, val_loader

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────────

    def _maybe_save_checkpoint(
        self, epoch: int, metric: float, metrics: dict[str, float]
    ) -> None:
        ckpt_cfg = self.cfg.checkpointing
        mode = self._monitor_mode
        improved = (
            (mode == "max" and metric > self.best_metric)
            or (mode == "min" and metric < self.best_metric)
        )

        if improved:
            self.best_metric = metric
            self.epochs_without_improvement = 0
            best_path = self.ckpt_dir / "best.pth"
            self._save_checkpoint(epoch, metrics, best_path)
            logger.info(
                f"  ✓ New best {self._monitor_key}={metric:.4f} → saved {best_path}"
            )
        else:
            self.epochs_without_improvement += 1

        if ckpt_cfg.get("save_last", True):
            self._save_checkpoint(epoch, metrics, self.ckpt_dir / "last.pth")

        # Top-k management
        top_k = ckpt_cfg.get("save_top_k", 3)
        if top_k > 0:
            epoch_path = self.ckpt_dir / f"epoch_{epoch:04d}.pth"
            self._save_checkpoint(epoch, metrics, epoch_path)
            self._prune_checkpoints(top_k)

    def _save_checkpoint(
        self, epoch: int, metrics: dict[str, float], path: Path
    ) -> None:
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_metric": self.best_metric,
                "metrics": metrics,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
            },
            path,
        )

    def _load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}. Starting from scratch.")
            return

        logger.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)

        model_target = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )
        model_target.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if ckpt.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if ckpt.get("scaler_state_dict") and self.scaler:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_metric = ckpt.get("best_metric", self.best_metric)

        logger.info(
            f"  Resumed at epoch {self.start_epoch}  "
            f"best_{self._monitor_key}={self.best_metric:.4f}"
        )

    def _prune_checkpoints(self, top_k: int) -> None:
        """Remove epoch checkpoints beyond the top-k most recent."""
        epoch_ckpts = sorted(self.ckpt_dir.glob("epoch_*.pth"))
        for old in epoch_ckpts[:-top_k]:
            old.unlink()

    # ──────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────

    def _log_metrics(self, metrics: dict[str, float], epoch: int, prefix: str) -> None:
        if self._writer is None:
            return
        for k, v in metrics.items():
            self._writer.add_scalar(f"{prefix}/{k}", v, epoch)

    # ──────────────────────────────────────────────────────────────────────
    # Early stopping
    # ──────────────────────────────────────────────────────────────────────

    def _should_stop_early(self) -> bool:
        es = self.cfg.training.get("early_stopping", {})
        if not es.get("enabled", False):
            return False
        patience = es.get("patience", 10)
        return self.epochs_without_improvement >= patience

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        hw = self.cfg.hardware
        requested = hw.get("device", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available — falling back to CPU.")
            return torch.device("cpu")
        if requested == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available — falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def _monitor_key(self) -> str:
        raw = self.cfg.checkpointing.get("monitor", "val/loss")
        # Strip the 'val/' prefix so it matches the keys in val_metrics dict
        return raw.split("/")[-1]      

    @property
    def _monitor_mode(self) -> str:
        return self.cfg.checkpointing.get("mode", "min")


# ---------------------------------------------------------------------------
# Linear warm-up wrapper
# ---------------------------------------------------------------------------

class _WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linearly ramps LR from 0 to base LR over `warmup_epochs`, then
    hands off to `base_scheduler`."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_scheduler,
        warmup_epochs: int,
        last_epoch: int = -1,
    ):
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * scale for base_lr in self.base_lrs]
        return self.base_scheduler.get_last_lr()

    def step(self, *args, **kwargs):
        if self.last_epoch >= self.warmup_epochs:
            self.base_scheduler.step(*args, **kwargs)
        super().step()

    def state_dict(self):
        return {
            "last_epoch": self.last_epoch,
            "base_scheduler": self.base_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.last_epoch = state["last_epoch"]
        self.base_scheduler.load_state_dict(state["base_scheduler"])


def get_class_from_string(dotted_path: str):
    """
    Resolves a dotted string like 'torch.nn.GELU' or 'nn.GELU' to the actual class.
    """
    # Handle shorthand like "nn.GELU" -> "torch.nn.GELU"
    if dotted_path.startswith("nn."):
        dotted_path = "torch." + dotted_path

    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def parse_model_config(model_cfg):
    # del model_cfg['architecture']
    param_dict = dict()
    for k,v in model_cfg.items():
        if k == 'architecture':
            continue
        if k == 'input':
            for subkey, subval in v.items():
                param_dict[subkey] = subval 
        elif k == 'act':
            param_dict[k] = get_class_from_string(v)
        else:
            param_dict[k] = v
    return param_dict