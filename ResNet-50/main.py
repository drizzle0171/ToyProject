import os
import tqdm
import random
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# from dataset import 
from model import ResNet50
from torchvision import transforms
import torchvision
import wandb

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])
batch_size = 128

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# model
net = ResNet50().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.15, momentum=0.9, weight_decay =0.0001)

# train
wandb.init(project="Toy-ResNet-50")
wandb.config = {
  "learning_rate": 0.1,
  "epochs": 20,
  "batch_size": 128
}

for epoch in range(30):
    epoch_loss = 0.0
    for i, data in enumerate(tqdm.tqdm(trainloader, desc=f'{epoch+1} epoch')):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    loss = epoch_loss/len(trainloader)
    wandb.log({"loss": loss})
    print(loss)
print('Finished Training')
torch.save(net, 'ResNet50.pt')

# validation
model = torch.load('ResNet50.pt')
model.eval()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.15, momentum=0.9, weight_decay =0.0001)

with torch.no_grad():
    epoch_loss=0
    for idx, data in enumerate(testloader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs).cuda()
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
    loss = epoch_loss/len(testloader)
    print(loss)
# validation: 1.158665125128589