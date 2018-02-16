from torch.autograd import Variable
from torch.utils.data import DataLoader
import models.multi_stream_model as multi_stream_model
import torch
import torchvision.transforms as transforms
import shutil

from util.function_library import AverageMeter, check_accuracy, check_whdr

from data.pair_wise_image import IIWDataset

# set cudnn to true.
# torch.backends.cudnn.enabled=True


def train(train_loader, model, criterion, optimizer, epoch):
    print("Start epoch number - ", epoch)

    model.train()

    for index, data in enumerate(train_loader):
        image, point_1_img, point_2_img, label, _ = data
        image, point_1_img, point_2_img, label = \
            Variable(image), Variable(point_1_img), Variable(point_2_img), Variable(label)

        if torch.cuda.is_available():
            image, point_1_img, point_2_img, label = image.cuda(), point_1_img.cuda(), point_2_img.cuda(), label.cuda()

        output = model(image, point_1_img, point_2_img)

        loss = criterion(output, label)

        # accuracy = check_accuracy(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(test_loader, model):
    model.eval()
    accuracy_meter, whdr_meter = AverageMeter(), AverageMeter()
    for index, data in enumerate(test_loader):
        image, point_1_img, point_2_img, label, weight = data
        image, point_1_img, point_2_img, label = \
            Variable(image), Variable(point_1_img), Variable(point_2_img), Variable(label)

        if torch.cuda.is_available():
            image, point_1_img, point_2_img = image.cuda(), point_1_img.cuda(), point_2_img.cuda()

        output = model(image, point_1_img, point_2_img).cpu()

        accuracy = check_accuracy(output, label)
        whdr_score = check_whdr(output, label, weight)
        whdr_meter.update(whdr_score)
        accuracy_meter.update(accuracy)

    model.train()
    return accuracy_meter.avg, whdr_meter.avg


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
train_loader = DataLoader(train_data_set, batch_size=128)

test_data_set = IIWDataset(mode='test', transforms=sim_transforms, resize_transform=resize_transform)
test_loader = DataLoader(test_data_set, batch_size=128)

if torch.cuda.is_available():
    model = multi_stream_model.MultiStreamNet().cuda()
    model = torch.nn.parallel.DataParallel(model)
    criterion = torch.nn.CrossEntropyLoss().cuda()
else:
    model = multi_stream_model.MultiStreamNet()
    criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=wt_decay)

for epoch in range(number_of_epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)

        accuracy, weighted_accuracy = evaluate(test_loader, model)

        is_best = accuracy > best_prec1
        best_prec1 = max(accuracy, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)