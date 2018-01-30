from torch.autograd import Variable
from .meter import *


def cudalize(state):
    inputs, targets = state["sample"]
    if state["engine"].cudable:
        inputs, targets = inputs.cuda(), targets.cuda()
    state.update(dict(sample=(inputs, targets)))


def variablize(state):
    inputs, targets = state["sample"]
    inputs, targets = Variable(inputs), Variable(targets)
    state.update(dict(sample=(inputs, targets)))


def network_specialize(state):
    if state["train"]:
        state["network"].train()
    else:
        state["network"].eval()


def loss_meter_initialize(state):
    state.update(dict(loss_meter=LossMeter()))


def acc_meter_initialize(state):
    state.update(dict(acc_meter=AccuracyMeter()))

def loss_meter_update(state):
    state["loss_meter"].add(state["loss"])


def acc_meter_update(state):
    _,targets = state["sample"]
    state["acc_meter"].add(targets,state["output"])


def report_progress(state):
    strs = []
    strs.append("Epoch={}".format(state["epoch"]))
    for name, item in state.items():
        if isinstance(item, Meter):
            strs = item.__str__()
    return " ".join(strs)
