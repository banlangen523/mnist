import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

import models


transformation = transforms.Compose([transforms.RandomAffine(10, (0.1, 0.1)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('data/', train=True, transform=transformation, download=True)
test_dataset = datasets.MNIST('data/', train=False, transform=transformation, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    
def fit(epoch, model, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
    
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available:
            data, target = data.to(device), target.to(device)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        running_loss += F.cross_entropy(output, target, reduction='sum').detach().item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        
        if phase == 'training':
            loss.backward()
            optimizer.step()
            
        del data, target, output, loss
    
    loss = running_loss/len(data_loader.dataset)
    print(f'{phase} loss is {loss:{5}.{2}}')
    return loss
    
model = models.Net()
if torch.cuda.is_available:
    model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
train_losses = []
val_losses = []

for epoch in range(1,40):
    epoch_loss = fit(epoch, model, train_loader, phase='training')
    val_epoch_loss = fit(epoch, model, test_loader, phase='validation')
    train_losses.append(epoch_loss)
    val_losses.append(val_epoch_loss)
    
plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label='training loss')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='valid loss')
plt.legend()

running_correct = 0
for data, target in test_loader:
    if torch.cuda.is_available():
        data, target = data.to(device), target.to(device)
    output = model(data)
    preds = output.data.max(dim=1, keepdim=True)[1]
    running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().item()

error_rate = 100. * (1 - running_correct/len(test_loader.dataset))
print(f'{error_rate:{10}.{4}}%')