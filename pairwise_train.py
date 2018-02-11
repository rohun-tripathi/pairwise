from torch.autograd import Variable
from torch.utils.data import DataLoader,
import models.multi_stream_model as multi_stream_model
import torch
import torchvision.transforms as transforms
import shutil

from data.pair_wise_image import IIWDataset

# set cudnn to true.
# torch.backends.cudnn.enabled=True

def train(train_loader, model, criterion, optimizer, epoch):
    pass


def validate(test_loader, model, criterion):
    pass


# TODO - presently we have no annealing.
def adjust_learning_rate(optimizer, epoch):
    ...


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Paper defined values
lr = 1e-3
wt_decay = 0.002
number_of_epochs = 15


# Standard ImageNet normalization.
resize_transform = transforms.Scale((150, 150))
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data_set = IIWDataset(mode='train', transforms=transforms, resize_transform=resize_transform)
train_loader = DataLoader(train_data_set, batch_size=128)

test_data_set = IIWDataset(mode='test', transforms=transforms, resize_transform=resize_transform)
test_loader = DataLoader(test_data_set, batch_size=128)

model = multi_stream_model.MultiStreamNet().cuda()
model = torch.nn.parallel.DataParallel(model)

# 3 label CrossEntropyLoss
criterion = torch.nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=wt_decay)


for epoch in range(number_of_epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)