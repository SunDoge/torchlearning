from .meter import *

import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DefaultClassificationEngine(object):
    def __init__(self, hooks={}):
        self.hooks = hooks
        self.hook("on_init", None)

        self.cudable = torch.cuda.is_available()

    def hook(self, name, state):
        if name in self.hooks:
            for f in self.hooks[name]:
                if state is None:
                    f()
                else:
                    f(state)

    def train(self, network, iterator, optimizer, epoch=0):
        state = {
            "engine": self,
            'network': network,
            'iterator': iterator,
            'optimizer': optimizer,
            'epoch': epoch,
            't': 0,
            'train': True,
        }

        self.hook('on_start', state)
        self.hook('on_start_epoch', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_sample', state)

            def closure():
                loss, output = state['network'](state['sample'])
                state['output'] = output
                state['loss'] = loss
                loss.backward()
                self.hook('on_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None
                return loss

            state['optimizer'].zero_grad()
            state['optimizer'].step(closure)
            self.hook('on_update', state)
            state['t'] += 1
        state['epoch'] += 1
        self.hook('on_end_epoch', state)

        self.hook('on_end', state)
        return state


    def validate(self, network, iterator):
        state = {
            "engine": self,
            'network': network,
            'iterator': iterator,
            't': 0,
            'train': False,
        }

        self.hook('on_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_sample', state)

            def closure():
                loss, output = state['network'](state['sample'])
                state['output'] = output
                state['loss'] = loss
                self.hook('on_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            closure()
            state['t'] += 1
        self.hook('on_end', state)
        return state


    def destroy(self):
        self.hook("on_destroy", None)
