import time
from torchlearning.meter import Meter

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
