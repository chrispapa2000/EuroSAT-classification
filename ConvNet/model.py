import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, act=nn.ReLU):
        super().__init__()
        
        stride = (1 if in_channels==out_channels else 2) # if the number of channels is increased we halve the spatial resolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            act()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            act()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ConvNet(nn.Module):
    def __init__(
        self,
        in_channels, 
        im_size, 
        num_classes, 
        channels=[64,64,64,128,128,128,128,256,256,256,256,256,256,512,512,512,512], 
        act=nn.ReLU):
        
        super().__init__()
        
        self.input_conv = nn.Conv2d(in_channels, channels[0], 7, 2, 2)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        prev_channels = channels[0]
        layers = []
        for n_channels in channels:
            layers.append(ConvLayer(prev_channels, n_channels, kernel_size=3, padding=1, act=act))
            prev_channels = n_channels
        self.layers = nn.ModuleList(layers)
            
        self.clf = nn.Linear(channels[-1], num_classes)
        
    
    def forward(self, x):
        x = self.input_conv(x)
        x = self.pool(x)
        
        for layer in self.layers:
            x = layer(x)
        
        # spatial average pooling
        features = x.mean(dim=(2,3))
        logits = self.clf(features)
        return logits