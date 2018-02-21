import torch
from torch.autograd import Variable

from util.function_library import AverageMeter, check_accuracy, print_and_log

import shutil


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    accuracy_meter = AverageMeter()

    training_loss = AverageMeter()
    for index, data in enumerate(train_loader):

        if index % 1000 == 0 and index > 0:
            print_and_log('Training', 'iter', epoch, 'batch_number', index, "loss", training_loss.avg)

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
        if index % 1000 == 0:
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
