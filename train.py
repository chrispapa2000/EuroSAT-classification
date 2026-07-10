from argparse import ArgumentParser

from Trainer import load_config
from Trainer.resnet_trainer import ResNetTrainer
from Trainer.convnet_trainer import ConvNetTrainer
from Trainer.vit_trainer import ViTTrainer
from Trainer.swin_trainer import SwinTrainer

def choose_trainer(config):
    if config.model.architecture == 'ResNet':
        trainer = ResNetTrainer(config)
    elif config.model.architecture == 'ConvNet':
        trainer = ConvNetTrainer(config)
    elif config.model.architecture == 'ViT':
        trainer = ViTTrainer(config) 
    elif config.model.architecture == 'Swin':
        trainer = SwinTrainer(config)
    else:
        raise NotImplementedError(f"Unknown model architecture: {config.model.architecture}") 
    return trainer

def main(args):
    conf = load_config(args.config)
    trainer = choose_trainer(conf)
    trainer.train()
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, 
                        # default='configs/config.yml'
                        default='configs/swin_config.yml'
    )
                        # required=True) 
    args = parser.parse_args()
    main(args)
    
    