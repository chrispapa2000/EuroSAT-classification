from argparse import ArgumentParser

from Trainer import load_config
from train import choose_trainer

import torch

def main(args):
    conf = load_config(args.config)
    trainer = choose_trainer(conf) 
    
    # init model
    trainer.model = trainer.build_model().to(trainer.device)
    
    # load best model
    model_path = trainer.ckpt_dir / 'best.pth'
    ckpt = torch.load(model_path, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.model.eval()
    
    # run test
    trainer._run_test_epoch()
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yml')
    args = parser.parse_args()
    main(args)
    