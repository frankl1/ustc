import numpy as np
from scipy.stats import norm

class UNumber:
    """ an uncertain number define by
    value: the best guest or the mean of the number
    err: the error or standard deviation of the number
    """
    SIMPLE_CMP = 'simple_cmp'
    CDF_CMP = 'cdf_cmp'
    INTERVAL_CMP = 'interval_cmp'
    def __init__(self, value, err, cmp_type, nb_cdf_sample=100):
        assert err >= 0, f'err must not be negative. Received {err}'
        self.value = UNumber.as_float32(value)
        self.err = UNumber.as_float32(err)
        self.cmp_type = cmp_type
        self.nb_cdf_sample = nb_cdf_sample
        
    @staticmethod
    def as_float32(value):
        floatinfo = np.finfo(np.float32)
        if value == np.inf or value > floatinfo.max:
            return floatinfo.max
        elif value == -np.inf or value < floatinfo.min:
            return floatinfo.min
        else:
            return value

    def __lt__(self, aUNumber):
        if self.cmp_type == UNumber.CDF_CMP:
            return self.cdf_lt(aUNumber)
        return self.simple_lt(aUNumber)
    
    def simple_lt(self, aUNumber):
        if (self.value < aUNumber.value) or (self.value == aUNumber.value and self.err < aUNumber.err):
            return True
        return False
    
    def cdf_lt(self, aUNumber):
        if self.err == 0 or aUNumber.err == 0:
            return self.simple_lt(aUNumber)

        lb = min(self.value - self.err, aUNumber.value - aUNumber.err)
        ub = max(self.value + self.err, aUNumber.value + aUNumber.err)
        pts = np.linspace(lb, ub, self.nb_cdf_sample)

        self_cdf = norm.cdf(pts, loc=self.value, scale = self.err)
        other_cdf = norm.cdf(pts, loc=aUNumber.value, scale=aUNumber.err)
        
        if np.isnan(self_cdf).any() or np.isnan(other_cdf).any():
            print(self_cdf, '\n', other_cdf, '\n', self, aUNumber)

        counts = np.sum(self_cdf > other_cdf)
        
        return True if counts > (self.nb_cdf_sample/2) else False
    
    def interval_lt(self, aUNumber):
        # the probability that self is less than aUNumber

        if aUNumber.err == 0 and self.err == 0:
            return self.value < aUNumber.value

        a_up = aUNumber.value + aUNumber.err
        a_lo = aUNumber.value - aUNumber.err
        la = a_up - a_lo

        b_up = self.value + self.err
        b_lo = self.value - self.err
        lb = b_up - b_lo
        
        p = max(1 - max((b_up - a_lo)/(la + lb), 0), 0)

        return p > 0.5

    def __str__(self):
        return f"{self.value}({self.err})"