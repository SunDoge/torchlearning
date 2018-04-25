from collections import defaultdict
import numbers
import json


class Recorder(defaultdict):
    def __init__(self, **kwargs):
        super(Recorder, self).__init__(list)
        self.update(kwargs)

    def add_scalar_summary(self, **kwargs):
        for key,value in kwargs.items():
            self[key].append(value)

    def __getattr__(self, attr):
        return self[attr]

    @property
    def json(self):
        return json.dumps(self, indent=2, )

    def dumps(self, filename):
        with open(filename, "w") as f:
            json.dump(self, f)

    @staticmethod
    def loads(filename):
        recorder = Recorder()
        with open(filename, "r") as f:
            recorder.update(json.load(f))
        return recorder


