import datetime
import logging

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import models.multi_stream_model as multi_stream_model
from data.pair_wise_image import IIWDataset
from network import train, update_learning_rate, evaluate, save_checkpoint

exp_name = "128_on_two_GPUS"
filename = 'logs/' + exp_name + '_experiment-{date:%d_%m_%H_%M_%S}.log'.format(date=datetime.datetime.now())
logging.basicConfig(filename=filename, level=logging.INFO)

# set cudnn to true.
# torch.backends.cudnn.enabled=True


def print_and_log(*inputs):
    if True:
        parts = "_".join([str(x) for x in inputs])
        print(parts)
        logging.info(parts)


# Paper defined values
lr = 1e-3
wt_decay = 0.002
max_steps = 50000
train_batch_size = 512


resize_transform = transforms.Scale((150, 150))
# Standard ImageNet normalization - required as we use pretrained ResNet on ImageNet.
sim_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data_set = IIWDataset(mode='train', transforms=sim_transforms, resize_transform=resize_transform)
train_loader = DataLoader(train_data_set, batch_size=train_batch_size, num_workers=8, shuffle=True)

test_data_set = IIWDataset(mode='test', transforms=sim_transforms, resize_transform=resize_transform)
test_loader = DataLoader(test_data_set, batch_size=1024, num_workers=8)

model = multi_stream_model.MultiStreamNet()
criterion = torch.nn.CrossEntropyLoss()

if torch.cuda.is_available():
    model = torch.nn.parallel.DataParallel(model).cuda()
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wt_decay)
best_prec1 = -1.0
step_count = 0
epoch = 0

load_best = False
if load_best:
    checkpoint = torch.load('trained_models/model_best.pth.tar')
    epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step_count = checkpoint['epoch'] * train_batch_size


while step_count < max_steps:
    update_learning_rate(optimizer, step_count, max_steps, train_batch_size)

    train(train_loader, model, criterion, optimizer, epoch)

    accuracy, weighted_err = evaluate(test_loader, model)

    print_and_log("Epoch", str(epoch), "val_accuracy", str(accuracy), "val_weighted_error", str(weighted_err))

    is_best = accuracy > best_prec1
    best_prec1 = max(accuracy, best_prec1)

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': 'resnet',
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)

    step_count += len(train_data_set) / train_batch_size
    epoch += 1
