import torch
import torch.nn as nn

# https://arxiv.org/pdf/1512.03385
# ResNet50 network structure
class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0 
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module): 
    # number of layer in each block for ResNet50: [3, 4, 6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride =1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride =2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride =2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride =2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)

        return(x)

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, 
                                                         kernel_size =1, stride = stride),
                                                nn.BatchNorm2d(out_channels*4))
        ###################################################################################
        # Why is downsampling needed?
        # In some cases, the main path changes the size or number of channels of the input 
        # feature map. For example:

        # Stride not equal to 1: The convolution operations in the main path might reduce 
        # the spatial dimensions 
        # (e.g., width and height) through the stride.

        # Change in the number of channels: The number of output channels in the main path
        # might differ from the number of input channels.

        # In these situations, the input on the skip connection path cannot be directly 
        # added to the output of the main path. Therefore, we need to use identity_downsample 
        # to adjust the size and the number of channels of the input to match the output of 
        # the main path.
        ###################################################################################

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4 # 64*4 = 256

        ###################################################################################
        # "If in_channels is 64, out_channels is 64, self.expansion is 4, and stride is 1:

        # identity_downsample will increase the number of input channels from 64 to 512 (128* 4) 
        # while also halving the spatial dimensions (due to stride=2)."
        ###################################################################################

        for i in range(num_residual_blocks -1):
            layers.append(block(self.in_channels, out_channels)) # 256 -> 64, 64*4 (256)again

        return nn.Sequential(*layers)

"""    
def ResNet50(img_channels=3, num_classes= 1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)
    
def ResNet101(img_channels=3, num_classes= 1000):
    return ResNet(block, [3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes= 1000):
    return ResNet(block, [3, 8, 36, 3], img_channels, num_classes)
"""