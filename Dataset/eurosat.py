import os
from zipfile import ZipFile
import rasterio
import numpy as np
from typing import Optional

from torch.utils.data import Dataset
import torch

class EuroSATDataset(Dataset):
    def __init__(
        self, 
        root: str, 
        transform: Optional[torch.nn.Module] = None,
        mode='train', 
        classes=('AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'), 
        normalize=True,
    ):
        super().__init__()
        self.root = root
        self.classes = classes
        self.mode = mode # train / val / test
        self.transform = transform
        
        self.class_dict = { cl: i for i, cl in enumerate(classes) }
        
        # pre-computed per-band stats, based on the trainset
        self.normalize = normalize
        self.mean = torch.tensor([1354.40546513, 1118.24399958, 1042.92983953,  947.62620298, 
                     1199.47283961, 1999.79090914, 2369.2229259 , 2296.82608341,
                     732.08340178,   12.11327804, 1819.01027862, 1118.92391149,
                     2594.14080791], dtype=torch.float)
        self.std = torch.tensor([245.7173245 ,  333.00767058,  395.09241013,  593.75049455, 
                    566.41690053,  861.18382767, 1086.63124889, 1117.98156753,
                    404.91975646,    4.77584468, 1002.58756276,  761.30315739,
                    1231.58568198], dtype=torch.float)
        
        self.extract_data()
        self.prepare_paths()
        
        
        
    def extract_data(self, ):
        data_name = 'EuroSAT_MS'
        data_dir = os.path.join(self.root, data_name)
        
        if os.path.exists(data_dir): # check if the data is already extracted
            pass
        else: # extract the data
            zippath = os.path.join(self.root, 'EuroSAT_MS.zip')
            with ZipFile(zippath) as Zobj:
                Zobj.extractall(self.root)
        
        # Check that all class directories are present
        subdirs = os.listdir(data_dir)
        assert len(subdirs) == len(self.classes), f"The number of data directories does not match the expected number of classes ({len(self.classes)})!"
        for cl in self.classes:
            assert cl in subdirs, f"No data directory was found for class {cl}!"
    
    def prepare_paths(self):
        split_path = os.path.join(self.root, 'splits', f"eurosat-{self.mode}.txt")
        with open(split_path) as f:
            lines = f.readlines()
        paths = []
        for line in lines:
            cl, num = line.split('.')[0].split('_')
            paths.append(os.path.join(self.root, 'EuroSAT_MS', cl, f"{cl}_{num}.tif"))
        self.paths = paths 
        
        targets = [item_path.split('/')[-1].split('_')[0] for item_path in paths]
        self.targets = [self.class_dict[target] for target in targets]
        
    def __getitem__(self, index):
        item_path = self.paths[index]
        
        # load data from disk
        with rasterio.open(item_path) as src:
            data = src.read().astype(np.float32)  # (13, 64, 64)
        
        data = torch.tensor(data, dtype=torch.float32)
        
        # optinal transforms
        if self.transform is not None:
            data = self.transform(data)
        
        # optional normalization
        if self.normalize:
            data = (data - self.mean[:, None, None]) / self.std[:, None, None]
        
        # get label
        # target = item_path.split('/')[-1].split('_')[0]
        target = self.targets[index]
        
        # return data, self.class_dict[target]
        return data, target
        
    
    def __len__(self):
        return len(self.paths)
    