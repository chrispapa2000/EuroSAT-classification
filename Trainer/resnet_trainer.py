"""
resnet_trainer.py
---------------------------------

Implements the four required abstract methods:
  build_datasets  →  torchvision ImageFolder datasets + transforms
  build_model     →  any torchvision timm model from cfg.model.architecture
  compute_loss    →  cross-entropy with optional label smoothing
  forward_step    →  standard (images, labels) batch

Everything else (AMP, grad clipping, checkpointing, TensorBoard, LR schedule,
early stopping, resume) is handled by BaseTrainer.

"""
import random
import time

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from Dataset.eurosat import EuroSATDataset

from Trainer import BaseTrainer, AverageMeter, parse_model_config, logger
from ResNet.model import ResNet

class Pad_And_Crop(nn.Module):
    def __init__(self, pad: int, crop: int, p: float = 0.5):
        super().__init__()
        self.pad = pad
        self.crop = crop
        self.transform = T.Compose(
            [T.Pad(self.pad),
            T.RandomResizedCrop(self.crop)]
        )
        self.p = p
    
    def forward(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img
    


class ResNetTrainer(BaseTrainer):
    """ResNet Trainer"""

    # ── 1. Datasets ──────────────────────────────────────────────────────

    def build_datasets(self) -> tuple[Dataset, Dataset]:
        """
        build eurosat dataset
        """
        data_cfg = self.cfg.data
        aug_cfg = data_cfg.augmentation
        norm_cfg = aug_cfg.normalize

        # normalize = T.Normalize(mean=norm_cfg.mean, std=norm_cfg.std)

        train_cfg = aug_cfg.train
        
        transforms = []
        if train_cfg.get("pad_and_crop", None) is not None:
            transforms += [Pad_And_Crop(train_cfg.pad_and_crop.pad, train_cfg.pad_and_crop.crop, train_cfg.pad_and_crop.p)] 
        transforms += [
            T.RandomHorizontalFlip() if train_cfg.get("random_horizontal_flip", True) else T.Lambda(lambda x: x),
            T.RandomVerticalFlip() if train_cfg.get("random_vertical_flip", True) else T.Lambda(lambda x: x),
        ]
        train_transforms = T.Compose(transforms=transforms)

        # val_cfg = aug_cfg.val
        val_transforms = None
        
        train_ds, val_ds = None, None
        if data_cfg.dataset == 'EuroSAT':
            train_ds = EuroSATDataset(root=data_cfg.root_dir, mode='train', transform=train_transforms, normalize=True)
            val_ds = EuroSATDataset(root=data_cfg.root_dir, mode='val', transform=val_transforms, normalize=True)
        else:
            raise NotImplementedError("not implemented dataset!") 

        return train_ds, val_ds

    # ── 2. Model ─────────────────────────────────────────────────────────

    def build_model(self) -> nn.Module:
        """
        Build ResNet model using custom implementation
        """
        model_cfg = self.cfg.model
        arch = model_cfg.architecture
        assert arch == "ResNet", "This is a ResNet trainer but the requested architecture is not ResNet!"
        
        param_dict = parse_model_config(model_cfg)
        
        model = ResNet(**param_dict)
        self.count_params(model)
        return model

    # ── 3. Loss ───────────────────────────────────────────────────────────

    def compute_loss(self, outputs, targets, batch) -> torch.Tensor:
        smoothing = self.cfg.training.get("label_smoothing", 0.0)
        return nn.CrossEntropyLoss(label_smoothing=smoothing)(outputs, targets)

    # ── 4. Forward step ───────────────────────────────────────────────────

    def forward_step(self, batch):
        images, labels = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        outputs = self.model(images)
        return outputs, labels
    

    # ── 5. Metrics (optional override) ────────────────────────────────────

    def compute_metrics(self, outputs, targets) -> dict[str, float]:
        metric_dict = dict()
        for metric in self.cfg.evaluation.metrics:
            if metric == 'acc': # top-1 accuracy
                with torch.no_grad():
                    acc = self._topk_accuracy(outputs, targets, k=1)
                    metric_dict[metric] = acc
            elif metric.startswith('acc@'): # case following acc@k pattern
                k = int(metric.split('@')[-1])
                with torch.no_grad():
                    acc = self._topk_accuracy(outputs, targets, k=k)
                    metric_dict[metric] = acc
        return metric_dict
    
    # ── 6. Training End ───────────────────────────────────────────────────
    
    def on_train_end(self):
        """
        Test trained model on test dataset
        """
        
        # # build test dataset and loader
        # data_cfg = self.cfg.data
        # test_transforms = None
        # test_ds = EuroSATDataset(root=data_cfg.root_dir, mode='test', transform=test_transforms, normalize=True)
        # test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, batch_size=self.cfg.data.loader.get("batch_size", 1))
        
        # # prepare for metric calculation
        # loss_meter = AverageMeter("loss")
        # metric_accumulators: dict[str, AverageMeter] = {}
        # t0 = time.time()
        
        # # load best model
        # model_path = self.ckpt_dir / 'best.pth'
        # ckpt = torch.load(model_path, map_location=self.device)
        # self.model.load_state_dict(ckpt["model_state_dict"])
        # self.model.eval()
        
        # with torch.no_grad():
        #     for batch_idx, batch in enumerate(test_loader):
        #         outputs, targets = self.forward_step(batch)
        #         loss = self.compute_loss(outputs, targets, batch)

        #         n = targets.size(0) if isinstance(targets, torch.Tensor) else 1
        #         loss_meter.update(loss.item(), n)

        #         # Metrics
        #         batch_metrics = self.compute_metrics(outputs, targets)
        #         for k, v in batch_metrics.items():
        #             if k not in metric_accumulators:
        #                 metric_accumulators[k] = AverageMeter(k)
        #             metric_accumulators[k].update(v, n)

        # elapsed = time.time() - t0
        # result = {"loss": loss_meter.avg}
        # result.update({k: m.avg for k, m in metric_accumulators.items()})

        # logger.info(
        #     f"End of Training [test]  "
        #     + "  ".join(f"{k}={v:.4f}" for k, v in result.items())
        #     + f"  ({elapsed:.1f}s)"
        # )
        # return result
        
        # load best model
        model_path = self.ckpt_dir / 'best.pth'
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # run test 
        self._run_test_epoch()
    
    def _run_test_epoch(self):
        logger.info("  → Running Evaluation on Test Set")
        # build test dataset and loader
        data_cfg = self.cfg.data
        test_transforms = None
        test_ds = EuroSATDataset(root=data_cfg.root_dir, mode='test', transform=test_transforms, normalize=True)
        test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, batch_size=self.cfg.data.loader.get("batch_size", 1))
        
        # prepare for metric calculation
        loss_meter = AverageMeter("loss")
        metric_accumulators: dict[str, AverageMeter] = {}
        t0 = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                outputs, targets = self.forward_step(batch)
                loss = self.compute_loss(outputs, targets, batch)

                n = targets.size(0) if isinstance(targets, torch.Tensor) else 1
                loss_meter.update(loss.item(), n)

                # Metrics
                batch_metrics = self.compute_metrics(outputs, targets)
                for k, v in batch_metrics.items():
                    if k not in metric_accumulators:
                        metric_accumulators[k] = AverageMeter(k)
                    metric_accumulators[k].update(v, n)

        elapsed = time.time() - t0
        result = {"loss": loss_meter.avg}
        result.update({k: m.avg for k, m in metric_accumulators.items()})

        logger.info(
            f"[test]  "
            + "  ".join(f"{k}={v:.4f}" for k, v in result.items())
            + f"  ({elapsed:.1f}s)"
        )
        return result
        
            
        
    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        with torch.no_grad():
            batch_size = targets.size(0)
            _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            correct_k = correct[:k].reshape(-1).float().sum()
            return (correct_k / batch_size).item()

    # @staticmethod
    # def _get_classifier_in_features(model: nn.Module) -> int:
    #     for attr in ("fc", "classifier", "head"):
    #         layer = getattr(model, attr, None)
    #         if layer is None:
    #             continue
    #         if isinstance(layer, nn.Linear):
    #             return layer.in_features
    #         if isinstance(layer, nn.Sequential):
    #             for sub in reversed(list(layer.children())):
    #                 if isinstance(sub, nn.Linear):
    #                     return sub.in_features
    #     raise ValueError("Cannot infer classifier in_features from model.")

    # @staticmethod
    # def _set_classifier(model: nn.Module, new_head: nn.Module) -> None:
    #     for attr in ("fc", "classifier", "head"):
    #         if hasattr(model, attr):
    #             setattr(model, attr, new_head)
    #             return
    #     raise ValueError("Cannot find classifier layer to replace.")

    # @staticmethod
    # def _freeze_backbone(model: nn.Module, freeze_until: str | None) -> None:
    #     freezing = True
    #     for name, param in model.named_parameters():
    #         if freeze_until and freeze_until in name:
    #             freezing = False
    #         if freezing:
    #             param.requires_grad_(False)


# # ---------------------------------------------------------------------------
# # CLI entry-point
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Image Classification Trainer")
#     parser.add_argument("--config", type=str, default="config.yaml")
#     args = parser.parse_args()

#     cfg = load_config(args.config)
#     trainer = ImageClassificationTrainer(cfg)
#     trainer.train()