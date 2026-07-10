import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange

from Dataset.eurosat import EuroSATDataset
from Trainer import load_config
from train import choose_trainer 

classes=('AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake')

colors = ('brown', 'red', 'chocolate', 'gold', 'olive', 'lawngreen', 'turquoise', 'deepskyblue', 'blueviolet', 'magenta')

def project_2d(embeddings, labels, algo='PCA', model='resnet'):
    # cast to numpy
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    algo = algo.upper()
    assert algo in ['PCA', 'TSNE'], "choose a dim reduction algo from PCA and TSNE"
    
    # dimensionality reduction  
    if algo == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
    elif algo == 'TSNE':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(embeddings)
    
    # save as image
    VIS_DIR = 'visualizations'
    os.makedirs(VIS_DIR, exist_ok=True)
    
    fig, ax = plt.subplots()
    x, y = embeddings_2d[:, 0], embeddings_2d[:, 1]
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(x[ix], y[ix], c=colors[g], label=classes[g])
    ax.legend()
    
    # ax.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=labels, )
    plt.title(f"Visualization of 2d projection with {algo}")
    plt.savefig(f"{VIS_DIR}/{model}_{algo}.png")
        

def get_embedding_vit(model, x):
    B = x.shape[0]
    # patchify input images
    x = model.patchifier(x)
    
    # apply positional encodings
    posemb = model.positional_encodings(torch.arange(model.sequence_len, device=x.device))
    x = x + posemb.unsqueeze(0)
    
    # prepend clf embedding
    clf_emb = model.clf_embedding[:, :].repeat((B,1))
    x = torch.cat([clf_emb.unsqueeze(1), x], dim=1)
    
    # pass through the transformer
    for layer in model.layers:
        x = layer(x)
    
    # only keep the clf embeddings
    x = x[:, 0, :]
    return x

def get_embedding_convolutional(model, x):
    x = model.input_conv(x)
    x = model.pool(x)
    
    for layer in model.layers:
        x = layer(x)
    
    # spatial average pooling
    features = x.mean(dim=(2,3))
    return features

def get_embedding_swin(model, x):
    B, C, H, W= x.shape
    assert H == model.H and W == model.W, f"Wrong input dimentions! Provide input with dimensions: (H,W)=({model.H}, {model.W})"
        
    # -- Patchify input images -- #
    x = model.patchifier(x)
    
    # -- Arange flattened patches into square grid -- #
    h, w = H//model.patch_size, W//model.patch_size
    x = rearrange(x, 'B (n1 n2) D -> B n1 n2 D', n1=h)
    
    # -- Apply Swin Transformer Stages -- #
    for stage in model.stages:
        x = stage(x)
    
    # -- Average Across Spatial dimensions -- #
    x_mean = x.mean(dim=(1,2)) 
    return x_mean



def choose_model(conf):
    # -- Prepare Model -- ##
    trainer = choose_trainer(conf) 
    
    # init model
    trainer.model = trainer.build_model().to(trainer.device)
    
    # load best model
    model_path = trainer.ckpt_dir / 'best.pth'
    ckpt = torch.load(model_path, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.model.eval()
    
    # -- Choose embedding extractor function depending on model -- #
    embedder = None
    if conf.model.architecture in ['ResNet', 'ConvNet']:
        embedder = get_embedding_convolutional
    elif conf.model.architecture == 'ViT':
        embedder = get_embedding_vit
    elif conf.model.architecture == 'Swin':
        embedder = get_embedding_swin
    else:
        raise NotImplementedError("Unknown model architecture!")
    
    return trainer, embedder
    
    
    
def main(args):
    # parse config
    conf = load_config(args.config)
    
    trainer, embedder = choose_model(conf)
    
    # dset = CIFAR10(root=args.data_root, train=False, transform=transforms)
    dset = EuroSATDataset(root=args.data_root, mode='test')
    global classes
    classes = dset.classes
    
    # pass all dataset images through the model and get clf_embeddings
    embeddings_list = []
    targets_list = []
    with torch.no_grad():
        for img, target in tqdm(dset):
            img = img.unsqueeze(0).to(trainer.device)
            x = embedder(trainer.model, img)
            embeddings_list.append(x.to('cpu').squeeze(0))
            targets_list.append(target)
    
    embeddings_list = np.array(embeddings_list)
    targets_list = np.array(targets_list)
    
    project_2d(embeddings_list, targets_list, 'PCA', trainer.cfg.model.architecture)
    project_2d(embeddings_list, targets_list, 'TSNE', trainer.cfg.model.architecture)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.yml')
    parser.add_argument('--data_root', default='./data/')
    args = parser.parse_args()
    main(args)