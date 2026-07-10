"""
convnet_trainer.py
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
import torch.nn as nn
import torchvision.transforms as T

from Trainer import parse_model_config
from Trainer.resnet_trainer import ResNetTrainer
from ConvNet.model import ConvNet

class ConvNetTrainer(ResNetTrainer):
    """ConvNet Trainer - same as ResNet Trainer except for build_model()"""

    def build_model(self) -> nn.Module:
        """
        Build ResNet model using custom implementation
        """
        model_cfg = self.cfg.model
        arch = model_cfg.architecture
        assert arch == "ConvNet", "This is a ConvNet trainer but the requested architecture is not ConvNet!"
        
        param_dict = parse_model_config(model_cfg)
        
        model = ConvNet(**param_dict)
        self.count_params(model)
        return model