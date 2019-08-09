"""
建模的主脚本
"""

import os
import logging.config

import pandas as pd

from model_base import return_feature, Manager

logging.config.fileConfig(r'D:\study\env\steam\log.ini')

LOG = logging.getLogger('my')

manager = Manager()

for method in [1, 2, 3]:
    manager.add_feature(str(method), return_feature(method))

for name in ['LinearRegression', 'LassoRegression', 'RidgeRegression']:
    manager.add_model(name)

manager.read_source_train()

manager.write_class_variable()
manager.read_class_variable()

print('manager.features \n', manager.features, '\n')
print('manager.models \n', manager.models, '\n')
print('manager.param_space \n', manager.param_space, '\n')
print('manager.source_train \n', manager.source_train, '\n')
print('manager.adjust_result \n', manager.adjust_result, '\n')
print('manager.validate_result \n', manager.validate_result, '\n')
print('manager.test_result \n', manager.test_result, '\n')
