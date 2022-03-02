#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch 
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
from tqdm import tqdm

### Import and transform data
transform = transforms.Compose([
    transforms.RandomResizedCrop(224,scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(1.0, 0.3, 0.3, 0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=100, 
                                          shuffle = True, num_workers = 1)

test_dataloader = torch.utils.data.DataLoader(testset, batch_size=100, 
                                          shuffle = True, num_workers = 1)


### Define Model
model = models.resnet18(pretrained=True)
# Disable grad for all conv layers
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, 10)
#using gpu or cpu 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


### Define Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


### Train
num_epochs=2
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        preds = model(inputs)
        loss = criterion(preds, labels)
        preds_class = preds.argmax(dim=1)
        loss.backward()
        optimizer.step()

        # print statistics(accuracy and loss)
        running_loss += loss.item()
        running_acc += (preds_class == labels.data).float().mean()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            print(f'accuracy: {running_acc / 10:.3f} %')
            running_loss = 0.0              
print('Finished Training')

### Validation
correct = 0
total = 0
model.eval()

for inputs, labels in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.set_grad_enabled(False):
        preds = model(inputs)
    
    _, predicted = torch.max(preds.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct // total} %')

