import torch


def check_accuracy(output, labels):
    batch_size = len(labels)
    _, pred = torch.max(output, 1)
    return float((pred == labels).sum().data.cpu().numpy()[0]) / batch_size


def check_whdr(output, labels, weights):
    _, pred = torch.max(output, 1)

    error_weight, total_weight = 0, 0
    for each_pred, each_label, each_weight in zip(pred.data.numpy(), labels.data.numpy(), weights):
        if each_pred != each_label:
            error_weight += each_weight
        total_weight += each_weight

    return (total_weight - error_weight)/total_weight


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