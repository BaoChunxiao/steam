"""
存放各种通用的小程序
"""

import time
import logging
from functools import wraps

LOG = logging.getLogger('my')


def timer(func):
    @wraps(func)
    def timer_wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        LOG.info(
            '%s,%s 运行时间: %d s' % (func.__module__, func.__name__, end - start))

    return timer_wrapper
