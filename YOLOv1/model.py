import torch
import torch.nn as nn
from config import config

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self,x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    

class YOLOv1(nn.Module):
    def __init__(self,in_channels=3,**kwargs):
        super().__init__()
        self.architecture = config.architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self,x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x,start_dim=1))

    def _create_fcs(self,split_size,num_boxes,num_classes):
        S,B,C = split_size,num_boxes,num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S,4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096,S*S*(C+B*5)) ##(S,S,30)
        )

    def _create_conv_layers(self,architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x,tuple):
                layers.append(
                    CNNBlock(
                        in_channels,
                        x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding = x[3]
                    )
                )
                in_channels = x[1]

            elif isinstance(x,str):
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
                
            elif isinstance(x,list):
                conv1,conv2,num_repeat = x

                for _ in range(num_repeat):
                    layers.append(
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding = conv1[3]
                        )
                    )
                    layers.append(
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    )
                    in_channels = conv2[1]

        return nn.Sequential(*layers)



if __name__ == '__main__':
    split_size,num_boxes,num_classes = 7,2,20
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = YOLOv1(split_size=split_size,num_boxes=num_boxes,num_classes=num_classes).to(device)

    x = torch.randn((2,3,448,448),device=device)

    output = model(x)

    print(output.shape)
    assert output.shape[1] == split_size**2 * (num_classes+5+5)