import torch


# Expects labels to be a list of integers
def check_accuracy(output, labels):
    batch_size = len(labels)
    _, pred = torch.max(output, 1)
    return float((pred == labels).sum().data.numpy()[0]) / batch_size


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count