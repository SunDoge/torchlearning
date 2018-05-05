import torch
from .meter import Meter

class Accuracy(object):
    def __init__(self, rate, n_correct, n_total):
        self.rate = rate
        self.n_correct = n_correct
        self.n_total = n_total

    def __str__(self):
        return f"Accuracy={self.rate*100:.4f}%({self.n_correct}/{self.n_total})"

class AccuracyMeter(Meter):
    def __init__(self, topk=(1,)):
        super(AccuracyMeter, self).__init__()
        self.topk = topk
        self.accuracies = None
        self.reset()

    def reset(self):
        self.accuracies = [Accuracy(rate=0.0, n_correct=0, n_total=0) for _ in self.topk]

    def add(self, targets, outputs):
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = targets.size(0)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            for accuracy, k in zip(self.accuracies, self.topk):
                accuracy.n_total += batch_size

                correct_k = correct[:k].view(-1).sum(0, keepdim=True).item()
                accuracy.n_correct += correct_k
                accuracy.rate = accuracy.n_correct / accuracy.n_total
                # res.append(correct_k.mul_(100.0 / batch_size))

    @property
    def value(self):
        return self.accuracies

    def __str__(self):
        accuracy_strs = []
        for k,acc in zip(self.topk,self.accuracies):
            accuracy_strs.append("Top-{} {}".format(k,acc))
        return "\n".join(accuracy_strs)