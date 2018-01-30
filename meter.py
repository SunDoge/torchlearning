import time
import math
import torch
from torch.autograd import Variable


class Meter(object):
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class TimeMeter(Meter):
    def __init__(self):
        super(TimeMeter, self).__init__()
        self.reset()

    def reset(self):
        self.n = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def __str__(self):
        duration_decimal,duration_integer = math.modf(self.value())
        strs = []
        if duration_integer > 24 * 60 * 60:
            days = duration_integer // (24 * 60 * 60)
            if days == 1:
                strs.append("1 Day")
            else:
                strs.append(f"{days} Days")
            duration_integer %= (24 * 60 * 60)

        if duration_integer > 60 * 60:
            hours = duration_integer // (60 * 60)
            if hours == 1:
                strs.append("1 Hour")
            else:
                strs.append(f"{hours} Hours")
            duration_integer %= (60 * 60)

        if duration_integer > 60:
            minutes = duration_integer // 60
            if minutes == 1:
                strs.append("1 Minute")
            else:
                strs.append(f"{minutes} Minutes")
            duration_integer %= 60

        if duration_integer <= 1:
            strs.append(f"{duration_integer+duration_decimal:.2f} Second")
        else:
            strs.append(f"{duration_integer+duration_decimal:.2f} Seconds")

        return " ".join(strs)


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