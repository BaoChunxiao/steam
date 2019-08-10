"""
建模的主脚本
"""

import logging.config

from model_base import return_feature, Manager

logging.config.fileConfig(r'D:\study\env\steam\log.ini')

LOG = logging.getLogger('my')

manager = Manager()

for method in [1, 2, 3]:
    manager.add_feature(str(method), return_feature(method))

manager.update_model('LinearRegression')
manager.update_model('LassoRegression')
manager.update_model('RidgeRegression')

manager.update_param_space('LinearRegression', {'normalize': [True, False]})
manager.update_param_space('LassoRegression', {'alpha': [0.5, 1, 2]})
manager.update_param_space('RidgeRegression', {'alpha': [0.5, 1, 2]})

manager.read_source_train()
manager.read_test_set()

manager.adjust(['1', '2', '3'],
               ['LinearRegression', 'LassoRegression', 'RidgeRegression'])

manager.validate(['1', '2', '3'],
                 ['LinearRegression', 'LassoRegression', 'RidgeRegression'])

manager.model_test(['2'],
                   ['RidgeRegression'])

manager.write_class_variable()
manager.read_class_variable()

print('manager.features \n', manager.features, '\n')
print('manager.models \n', manager.models, '\n')
print('manager.param_spaces \n', manager.param_spaces, '\n')
print('manager.source_train \n', manager.source_train, '\n')
print('manager.adjust_result \n', manager.adjust_result, '\n')
print('manager.validate_result \n', manager.validate_result, '\n')
print('manager.test_result \n', manager.test_result, '\n')
