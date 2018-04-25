import torch
from torchlearning.meter import Meter
from torch.autograd import Variable

class AccuracyMeter(Meter):
    def __init__(self,top=1):
        super(AccuracyMeter, self).__init__()

        self.reset()

    def reset(self):
        self.total = 0
        self.correct = 0

    def add(self, targets, outputs):
        targets = targets.data if isinstance(targets, Variable) else targets
        outputs = outputs.data if isinstance(outputs, Variable) else outputs

        _, predicted = torch.max(outputs, 1)
        self.total += targets.numel()
        self.correct += predicted.eq(targets).cpu().sum()

    def value(self):
        return self.correct / self.total

    def __str__(self):
        return "Accuracy={:.2f}".format(100*self.value())
