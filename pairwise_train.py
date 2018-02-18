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


def print_and_log(*inputs):
    if True:
        parts = "_".join([str(x) for x in inputs])
        print(parts)
        logging.info(parts)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    accuracy_meter = AverageMeter()

    training_loss = AverageMeter()
    for index, data in enumerate(train_loader):

        if index % 1000 == 0 and index > 0:
            print_and_log('Training epoch, iter- ', epoch, index, "loss", training_loss.avg)

        image, point_1_img, point_2_img, label, _ = data
        image, point_1_img, point_2_img, label = \
            Variable(image), Variable(point_1_img), Variable(point_2_img), Variable(label)

        if torch.cuda.is_available():
            image, point_1_img, point_2_img, label = image.cuda(), point_1_img.cuda(), point_2_img.cuda(), label.cuda()

        output = model(image, point_1_img, point_2_img)

        loss = criterion(output, label)
        training_loss.update(loss.data[0], image.size(0))

        accuracy = check_accuracy(output, label)
        accuracy_meter.update(accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print_and_log("Epoch", str(epoch), "train_accuracy", str(accuracy_meter.avg))


def evaluate(test_loader, model):
    model.eval()
    accuracy_meter, whdr_err_meter = AverageMeter(), AverageMeter()
    number_of_batches = len(test_loader)

    for index, data in enumerate(test_loader):
        if index % 1 == 0:
            print_and_log("Eval", "batch_number", index, "total", number_of_batches)

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
def update_learning_rate(optimizer, step_count, max_steps, bs):
    ...


def save_checkpoint(state, is_best, filename='trained_models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'trained_models/model_best.pth.tar')


# Paper defined values
lr = 1e-3
wt_decay = 0.002
max_steps = 20000
train_batch_size=128


resize_transform = transforms.Scale((150, 150))
# Standard ImageNet normalization - required as we use pretrained ResNet on ImageNet.
sim_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data_set = IIWDataset(mode='train', transforms=sim_transforms, resize_transform=resize_transform)
train_loader = DataLoader(train_data_set, batch_size=train_batch_size, num_workers=8, shuffle=True)

test_data_set = IIWDataset(mode='test', transforms=sim_transforms, resize_transform=resize_transform)
test_loader = DataLoader(test_data_set, batch_size=512, num_workers=8)

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
