import torch.nn as nn
import torch.nn.functional as F
import torch

class PNet(nn.Module):
    def __init__(self):
        super(PNet,self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,stride=1,padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(10,16,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=1),
            nn.PReLU()
        )
        #detection
        self.conv4_1 = nn.Conv2d(32,1,kernel_size=1,stride=1)
        #bounding box regression
        self.conv4_2 = nn.Conv2d(32,4,kernel_size=1,stride=1)
        #allignment
        self.conv4_3 = nn.Conv2d(32,10,kernel_size=1,stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        label = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        alli = self.conv4_3(x)
        return label,offset,alli


class RNet(nn.Module):
    def __init__(self):
        super(RNet,self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1,padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU()
        )
        self.conv4 = nn.Linear(64*3*3,128)
        self.prelu4 = nn.PReLU()
        #detection
        self.conv5_1 = nn.Linear(128,1)
        #bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        #allignment
        self.conv5_3 = nn.Linear(128,10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0),-1)
        x = self.conv4(x)
        x = self.prelu4(x)
        label = torch.sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)
        alli = self.conv5_3(x)
        return label, offset, alli


class ONet(nn.Module):
    def __init__(self):
        super(ONet,self).__init__()
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1,padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU()
        )
        self.conv5 = nn.Linear(128 * 3 * 3, 256)
        self.prelu5 = nn.PReLU()
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        #allignment
        self.conv6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        label = torch.sigmoid(self.conv6_1(x))
        offset = self.conv6_2(x)
        alli = self.conv6_3(x)
        return label,offset,alli