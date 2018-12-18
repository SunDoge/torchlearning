class LargerMetric(object):
    def __init__(self, initial_value=float("-inf")):
        self.current_value = initial_value

    def __call__(self, value):
        if value > self.current_value:
            self.current_value = value
            return True
        else:
            return False


class SmallerMetric(object):
    def __init__(self, initial_value=float("inf")):
        self.current_value = initial_value

    def __call__(self, value):
        if value < self.current_value:
            self.current_value = value
            return True
        else:
            return False
