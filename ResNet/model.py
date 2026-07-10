import torch.nn as nn

class ResidualLayer(nn.Module):
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
        
        self.proj_shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.proj_shortcut(residual)
        x = x + residual
        return x

class ResNet(nn.Module):
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
            layers.append(ResidualLayer(prev_channels, n_channels, kernel_size=3, padding=1, act=act))
            prev_channels = n_channels
        self.layers = nn.ModuleList(layers)
        
        # # calculate output spatial dim
        # output_size = 1 + (im_size + 2*2 - 7) // 2 # input conv
        # output_size = 1 + (output_size + 2*1 - 3) // 2 # pooling layer
        # for _ in range(len(set(channels))-1):
        #     output_size = 1 + (output_size + 2*1 - 3) // 2 # account for the spatial downsampling performed in layers with increased number of channels
            
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