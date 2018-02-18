from torch.autograd import Variable
from torch.utils.data import DataLoader
import models.multi_stream_model as multi_stream_model
import torch
import torchvision.transforms as transforms
import shutil

from util.function_library import AverageMeter, check_accuracy, check_whdr

from data.pair_wise_image import IIWDataset

import logging

import datetime

exp_name = "128_on_two_GPUS"
filename = 'logs/' + exp_name + '_experiment-{date:%d_%m_%H_%M_%S}.log'.format(date=datetime.datetime.now())
logging.basicConfig(filename=filename, level=logging.INFO)

# set cudnn to true.
# torch.backends.cudnn.enabled=True


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    accuracy_meter = AverageMeter()
    number_of_batches = len(train_loader)
    for index, data in enumerate(train_loader):

        if index % 10 == 0:
            print('Training epoch number- ', epoch, index, number_of_batches)
            print("_".join(["Epoch", str(epoch), "train_accuracy", str(accuracy_meter.avg)]))

        image, point_1_img, point_2_img, label, _ = data
        image, point_1_img, point_2_img, label = \
            Variable(image), Variable(point_1_img), Variable(point_2_img), Variable(label)

        if torch.cuda.is_available():
            image, point_1_img, point_2_img, label = image.cuda(), point_1_img.cuda(), point_2_img.cuda(), label.cuda()

        output = model(image, point_1_img, point_2_img)

        loss = criterion(output, label)

        accuracy = check_accuracy(output, label)
        accuracy_meter.update(accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("_".join(["Epoch", str(epoch), "train_accuracy", str(accuracy_meter.avg)]))


def evaluate(test_loader, model):
    model.eval()
    accuracy_meter, whdr_err_meter = AverageMeter(), AverageMeter()
    number_of_batches = len(test_loader)

    for index, data in enumerate(test_loader):
        if index % 1000 == 0:
            print("Eval", "batch_number", index, "total", number_of_batches)

        image, point_1_img, point_2_img, label, weight = data
        image, point_1_img, point_2_img, label = \
            Variable(image, volatile=True), Variable(point_1_img, volatile=True), Variable(point_2_img, volatile=True),\
            Variable(label, volatile=True)

        if torch.cuda.is_available():
            image, point_1_img, point_2_img = image.cuda(), point_1_img.cuda(), point_2_img.cuda()

        output = model(image, point_1_img, point_2_img).cpu()

        accuracy = check_accuracy(output, label)
        # whdr_error = check_whdr(output, label, weight)
        # whdr_err_meter.update(whdr_error)
        accuracy_meter.update(accuracy)

    model.train()
    return accuracy_meter.avg, whdr_err_meter.avg


# Presently we have no annealing.
def adjust_learning_rate(optimizer, epoch):
    ...


def save_checkpoint(state, is_best, filename='trained_models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'trained_models/model_best.pth.tar')


# Paper defined values
lr = 1e-3
wt_decay = 0.002
number_of_epochs = 15


resize_transform = transforms.Scale((150, 150))
# Standard ImageNet normalization - required as we use pretrained ResNet on ImageNet.
sim_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data_set = IIWDataset(mode='train', transforms=sim_transforms, resize_transform=resize_transform)
train_loader = DataLoader(train_data_set, batch_size=64, num_workers=8)

test_data_set = IIWDataset(mode='test', transforms=sim_transforms, resize_transform=resize_transform)
test_loader = DataLoader(test_data_set, batch_size=512, num_workers=8)

model = multi_stream_model.MultiStreamNet()

criterion = torch.nn.CrossEntropyLoss()

if torch.cuda.is_available():
    model = torch.nn.parallel.DataParallel(model).cuda()
    criterion = criterion.cuda()

optimizer = torch.optim.Adagrad(model.parameters(), lr)

best_prec1 = -1.0
for epoch in range(number_of_epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)

        accuracy, weighted_err = evaluate(test_loader, model)

        result = "_".join(["Epoch", str(epoch), "val_accuracy", str(accuracy), "val_weighted_error", str(weighted_err)])
        logging.info(result)
        print(result)

        is_best = accuracy > best_prec1
        best_prec1 = max(accuracy, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)