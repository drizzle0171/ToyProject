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
from torch.utils.data import DataLoader, Subset
import torchvision
import wandb
from sklearn.model_selection import train_test_split
import torch.distributed as distributed
from torch.multiprocessing import Process

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
batch_size = 128

# data load
train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# data split
train_indices, val_indices, _, _ = train_test_split(range(len(train)), train.targets, test_size=.1,)
train_set = Subset(train, train_indices)
val_set = Subset(train, val_indices)

# dataloader
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
valloder = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# model
model = ResNet50().cuda()
num_parmas = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The Number of params of model: ', num_parmas)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.15, momentum=0.9, weight_decay =1e-4)

decay_epoch = [32000, 48000]
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)

# train
wandb.init(project="Toy-ResNet-50")
wandb.config = {
  "learning_rate": 0.1,
  "epochs": 182,
  "batch_size": 128
}

steps = 0
for epoch in range(182):
    epoch_loss = 0.0
    for i, data in enumerate(tqdm.tqdm(trainloader, desc=f'{epoch+1} epoch')):
        steps+=1
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        schedular.step()
        epoch_loss += loss.item()
        if steps == 64001:
            break
    if steps == 64001:
        break
    loss = epoch_loss/len(trainloader)
    wandb.log({"loss": loss})
    print('Training loss: ', loss)
    print('Steps: ', steps)
print('Finished Training')
# torch.save(net, 'ResNet50.pt')

# validation
# model = torch.load('ResNet50.pt')
model.eval()
with torch.no_grad():
    epoch_loss=0
    for idx, data in enumerate(tqdm.tqdm(valloder, desc=f'{epoch+1} epoch')):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs).cuda()
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
    loss = epoch_loss/len(valloder)
    print(loss)

# validation(20 epochs): 1.158665125128589
# validation(500 epochs): 0.7034666799008846
# validation(64000 steps): 