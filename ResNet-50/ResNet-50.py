import torch
from torch import nn

##################################################### Data Loader ########################################################

# 제주도 다녀와서 하기







##########################################################################################################################


###################################################### model #############################################################

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4 # channel이 3x3을 거치고 4배가 됨

        # 1x1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels) # BatchNorm은 channel에 따라 계산

        # 3x3 -> 손실 정보를 없애기 위해서 padding = 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=stride, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        # relu 정의
        self.relu = nn.ReLU()

        # identity mapping: shape(dim)을 맞춰주기 위함 
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        # 1x1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 3x3
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # 1x1
        x = self.conv3(x)
        x = self.bn3(x)

        # identity
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity

        # relu
        x = self.relu(x)
        return x

# ResNet 50 Architecture: 3 4 6 3
class ResNet(nn.Module): 
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # layer_name: conv1 (output: 112x112x64)
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3) # padding = 3이어야지 output = 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # layer_name: conv2x (output: 56x56x256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #(output: 56x56x64)
        self.layer1 = self.layer(block, layers[0], out_channels=64, stride=1)

        # layer_name: conv3x (output: 28x28x512)
        self.layer2 = self.layer(block, layers[1], out_channels=128, stride=2)

        # layer_name: conv4x (output: 14x14x1024)
        self.layer3 = self.layer(block, layers[2], out_channels=256, stride=2)

        # layer_name: conv5x (output: 7x7x2048)
        self.layer4 = self.layer(block, layers[3], out_channels=512, stride=2)

        # average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    # 50개의 layer를 다 쌓을 수 없으니...
    def layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != self.out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet 전
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet 중
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ResNet 후
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # Linear layer에 넣어주기 위해 reshape
        y = self.fc(x)

        return y

# 드디어 ResNet
def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3,4,6,3], img_channels, num_classes)

#########################################################################################################################

##################################################### Validation ########################################################








#########################################################################################################################