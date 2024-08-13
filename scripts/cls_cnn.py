import torch
import torch.nn as nn

# CNN network structure
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(    # input (1,224,224)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2 # if stride=1, padding=(kernel_size-1)/2 = (5-1)/2=2
            ), # -> (16,224,224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # ->(16,112,112)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2), # ->(32,112,112)  
            nn.ReLU(),
            nn.MaxPool2d(2) # ->(32,56,56)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*56*56,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   # (batch, 32,56,56)
        x = x.view(x.size(0), -1) # (batch, 32*56*56)
        output = self.fc(x)
        return output