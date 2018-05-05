import torch
from .meter import Meter


class LossMeter(Meter):
    def __init__(self):
        super(LossMeter, self).__init__()
        self.reset()

    def reset(self):
        self.n = 0
        self.loss = 0.

    def add(self, loss):
        if torch.is_tensor(loss):
            self.n += 1
            self.loss += loss.item()
        else:
            raise ValueError("'loss' should be torch.tensor(scalar), but found {}"
                             .format(type(loss)))

    @property
    def value(self):
        if self.n == 0:
            return 0.0
        return self.loss / self.n

    @property
    def record(self):
        return dict(loss=self.value)

    def __str__(self):
        return "Loss={:.4f}".format(self.value())