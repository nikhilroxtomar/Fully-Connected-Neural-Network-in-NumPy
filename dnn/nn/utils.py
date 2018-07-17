## various utility functions

import cupy as cp

def _xW(x, W):
    """
    param: x - icput value
    param: W - weight

    if x=(m, n) W = (p, q), then
        n == p
    """
    assert x.shape[1] == W.shape[0]
    return cp.dot(x, W)


## Time function
def _time(total_time):
    if total_time < 60:
        return "{:0.3f} seconds".format(total_time)
    elif total_time > 60:
        return "{:0.3f} minutes".format(total_time/60.0)
