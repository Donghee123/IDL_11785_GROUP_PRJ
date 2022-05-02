'''Train CIFAR10 with PyTorch.'''
import os

import torch
import torch.nn as nn
import torch.optim as optim

import torch.backends.cudnn as cudnn

import torchvision

from utils import *
from tqdm import tqdm
import argparse



# Training
def train(net, epoch, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    batch_bar = tqdm(total=len(trainloader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        batch_bar.set_postfix(acc="{:.04f}%".format(100.*correct/total), loss="{:.04f}".format((train_loss/(batch_idx+1)), num_correct=correct, num_total = total))
        batch_bar.update() 
    batch_bar.close() 


def test(net, epoch, strDefenceName, testloader, criterion, isSave = True):
    best_acc = 0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        batch_bar = tqdm(total=len(testloader), dynamic_ncols=True, leave=False, position=0, desc='Test') 
        for batch_idx, (inputs, targets) in enumerate(testloader):            
            inputs, targets = inputs.to(device), targets.to(device)
            

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_bar.set_postfix(acc="{:.04f}%".format(100.*correct/total), loss="{:.04f}".format((test_loss/(batch_idx+1)), num_correct=correct, num_total = total))
            batch_bar.update() 
        batch_bar.close() 
    # Save checkpoint.
    acc = 100.*correct/total
    print(f'test acc: {acc}')
    if isSave == True:
        if acc > best_acc:
            print(f'Saving..{acc}')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('models'):
                os.mkdir('models')
            if not os.path.isdir(f'./models/{strDefenceName}'):
                os.mkdir(f'./models/{strDefenceName}')
            torch.save(state, f'./models/{strDefenceName}/{acc:.2f}.pth')
            best_acc = acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model Train code (None, Aug, FS, SPS)')

    parser.add_argument("-defence_name", type=str, help='Defence name only support (None, Aug, FS, SPS)', default='Aug')
    parser.add_argument("-lr", type=float, help='input learning rate', default= 0.1)
    
    args = parser.parse_args()
    
    defence_name = args.defence_name
    resume = False
    lr = args.lr

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_acc = 0  
    start_epoch = 0  

    # Data
    print('==> Preparing data..')

    transform_train, transform_test = get_train_test_transfomes(defence_name)

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')

    net = get_model(defence_name)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('/content/drive/MyDrive/CMU/IDL/Team_project/checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch+200):
        train(net, epoch, criterion, optimizer)
        test(net, epoch, defence_name, criterion)
        scheduler.step()

