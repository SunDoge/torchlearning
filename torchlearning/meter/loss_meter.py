import torch
from torchlearning.meter import Meter
from torch.autograd import Variable


class LossMeter(Meter):
    def __init__(self):
        super(LossMeter, self).__init__()
        self.reset()

    def reset(self):
        self.n = 0
        self.loss = 0.

    def add(self, loss):
        if isinstance(loss, Variable):
            loss = loss.data
        if torch.is_tensor(loss):
            self.n += 1
            self.loss += loss[0]
        else:
            raise Exception("Bad loss.")

    def value(self):
        return self.loss / self.n

    def __str__(self):
        return "Loss={:.4f}".format(self.value())