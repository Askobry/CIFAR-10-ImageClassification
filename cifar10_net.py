import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models.cnn_vit import cnn_ViT
from models.LeNet import LeNet
from models.resnet import ResNet
from models.vgg import VGG
from models.vit import ViT
from utils import *


# random seed set, in order to repeat
def seed_all(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)

seed_all(1873)

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='LeNet')
parser.add_argument('--bs', default='64')
parser.add_argument('--n_epochs', type=int, default='10')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--cos', action='store_true', help='Train with cosine annealing scheduling')
parser.add_argument('--use_tensorboard', default=True)
args = parser.parse_args()

if args.cos:
    from warmup_scheduler import GradualWarmupScheduler
if args.aug:
    import albumentations

batch_size = int(args.bs)

# tensorboard
if args.use_tensorboard:
    try:
        from torch.utils.tensorboard import SummaryWriter  
    except ImportError:
        from tensorboardX import SummaryWriter
    summary_writer = SummaryWriter(log_dir=os.path.join("tensorboard-output/", args.net)) # 每个模型对应一个文件夹
else:
    summary_writer = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# 1. Load and normalize CIFAR10
print('==> Loading data...')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                        download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                       download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# 2. Build model
print('==> Building model...')
if args.net[:6]=='resnet':
    net = ResNet(args.net)
elif args.net[:3]=='vgg':
    net = VGG(args.net)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
        image_size = 32,
        patch_size = args.patch,
        num_classes = 10,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
elif args.net=="cnn_vit":
    net = cnn_ViT(
        image_size=32,
        patch_size=args.patch,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
else:
    net = LeNet()

p_number = network_parameters(net)
net = net.to(device)
torch.backends.cudnn.benchmark = True
'''
A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
'''

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.net+'_latest.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# 3. Define a Loss function and optimizer
# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  

# reduce LR on Plateau
if not args.cos:
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-6, factor=0.1)
else:
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Network:            {args.net}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(args.n_epochs)}
    Batch sizes:        {args.bs}
    Learning rate:      {args.lr}
    GPU:                {device}''')
print('------------------------------------------------------------------')

# 4. Train the network
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
 
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
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
    
    loss_item = train_loss/(batch_idx+1)            
    if summary_writer:
        summary_writer.add_scalar('losses/train_loss', loss_item, global_step=epoch)
        summary_writer.add_scalar('acc/train_acc', 100.*correct/total, global_step=epoch)
    return loss_item

# 5. Test the network on the test data
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)
    
    # Save best acc pth.
    acc = 100.*correct/total
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'_best.pth')
        best_acc = acc
    
    # Save last epoch
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+args.net+'_latest.pth')

    # os.makedirs("log", exist_ok=True)
    # content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    # with open(f'log/log_{args.net}.txt', 'a') as appender:
    #     appender.write(content + "\n")
    
    if summary_writer:
        summary_writer.add_scalar('losses/val_loss', test_loss, global_step=epoch)
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        summary_writer.add_scalar('acc/val_acc', acc, global_step=epoch)
        
    return test_loss, acc

list_loss = []
list_acc = []
total_start_time = time.time()
for epoch in range(start_epoch, args.n_epochs):
    epoch_start_time = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    print("Time: {:.4f}\tAcc: {:.5f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(time.time() - epoch_start_time,
                                                                                acc, val_loss, optimizer.param_groups[0]["lr"]))
    if args.cos:
        scheduler.step(epoch - 1)

    list_loss.append(val_loss)
    list_acc.append(acc)

    # write as csv for analysis
    # with open(f'log/log_{args.net}.csv', 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerow(list_loss)
    #     writer.writerow(list_acc)
    # print(list_loss)
summary_writer.close()
total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
