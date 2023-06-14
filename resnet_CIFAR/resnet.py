import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResidualBlcok(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,shortcut=None)-> None:
        super(ResidualBlcok,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.right = shortcut

    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out +=residual
        return F.relu(out)
    
class ResNet(nn.Module):
    def __init__(self,num_classes =10) -> None:
        super(ResNet,self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1,64,3,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1 =self._make_layers(64,128,3)
        # self.layer2 =self._make_layers(128,256,4,stride=2)
        # self.layer3 =self._make_layers(256,512,6,stride=2)
        # self.layer4 =self._make_layers(512,512,3,stride=2)
        self.fc = nn.Linear(512,num_classes)


    def _make_layers(self,in_channel,out_channel,block_num,stride = 1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,1,stride,bias=False),
            nn.BatchNorm2d(out_channel)
        )
        layers = []
        layers.append(ResidualBlcok(in_channel,out_channel,stride,shortcut))
        for i in range(1,block_num):
            layers.append(ResidualBlcok(out_channel,out_channel))
        #starred expression takes  every element out of the list
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.pre(x)
        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = F.avg_pool2d(x,3)
        x = x.view(x.size()[0],-1)
        return self.fc(x)
    
if __name__ =='__main__':
    model = ResNet()
