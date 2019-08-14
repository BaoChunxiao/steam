"""
建模的主脚本
"""

import logging.config

import numpy as np

from model_main import Manager

logging.config.fileConfig(r'D:\study\env\steam\log.ini')

LOG = logging.getLogger('my')


def add_feature():
    for method in [1, 2, 3]:
        manager.add_feature(str(method), manager.return_feature(method))


def update_model():
    manager.update_model('LinearRegression')
    manager.update_model('LassoRegression')
    manager.update_model('RidgeRegression')
    manager.update_model('DecisionTreeRegressor')
    manager.update_model('GradientBoostingRegressor',
                         min_samples_leaf=5, n_iter_no_change=5)
    manager.update_model('RandomForestRegressor', min_samples_leaf=5)
    manager.update_model('SVR', gamma='scale')


def update_param_space():
    manager.update_param_space('LinearRegression', {
        'normalize': [True, False]
    })

    manager.update_param_space('LassoRegression', {
        'alpha': [round(i, 1) for i in np.linspace(0.1, 50, 100)]
    })

    manager.update_param_space('RidgeRegression', {
        'alpha': [round(i, 1) for i in np.linspace(0.1, 50, 100)]
    })

    manager.update_param_space('DecisionTreeRegressor', {
        'min_samples_leaf': [int(i) for i in np.linspace(1, 100, 100)],
    })

    manager.update_param_space('GradientBoostingRegressor', {
        'n_estimators': [500],
        'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.7],
        'max_features': [0.7],
    })

    manager.update_param_space('RandomForestRegressor', {
        'n_estimators': [300, 400, 500],
        'max_depth': [3, 4, 5, 6],
        'max_features': [0.8, 0.7, 0.6, 0.5],
    })

    manager.update_param_space('SVR', {
        'kernel': ['linear', 'poly'],
        'C': [round(i, 1) for i in np.linspace(0.1, 2, 20)]
    })


if __name__ == '__main__':

    manager = Manager()

    manager.read_class_variable()

    # add_feature()
    # update_model()
    # update_param_space()

    # manager.read_source_train()
    # manager.read_test_set()
    # manager.create_groups()
    # manager.create_stacking_groups()

    manager.adjust(
        ['2', '3'],
        ['LinearRegression','LassoRegression']
    )

    # ['LinearRegression', 'LassoRegression', 'RidgeRegression',
    #  'DecisionTreeRegressor', 'GradientBoostingRegressor',
    #  'RandomForestRegressor', 'SVR']

    manager.model_validate(
        ['2', '3'],
        ['LinearRegression', 'LassoRegression']
    )

    manager.merge_validate(
        ['2', '3'],
        ['Stacking'],
        ['LinearRegression', 'LassoRegression', 'RidgeRegression',
         'DecisionTreeRegressor', 'GradientBoostingRegressor',
         'RandomForestRegressor', 'SVR'],
        'RandomForestRegressor'
    )

    manager.model_test(
        ['2', '3'],
        ['LinearRegression', 'LassoRegression']
    )

    manager.merge_test(
        ['2', '3'],
        ['Stacking'],
        ['LinearRegression', 'LassoRegression', 'RidgeRegression',
         'DecisionTreeRegressor', 'GradientBoostingRegressor',
         'RandomForestRegressor', 'SVR'],
        'SVR'
    )

    manager.write_class_variable()

    print('manager.features \n', manager.features, '\n')
    print('manager.models \n', manager.models, '\n')
    print('manager.param_spaces \n', manager.param_spaces, '\n')
    print('manager.source_train \n', manager.source_train, '\n')
    print('manager.test_set \n', manager.test_set, '\n')
    print('manager.groups \n', manager.groups, '\n')
    print('manager.cv_predict \n', manager.cv_predict, '\n')
    print('manager.stacking_groups \n', manager.stacking_groups, '\n')
    print('manager.adjust_result \n', manager.adjust_result, '\n')
    print('manager.validate_result \n', manager.validate_result, '\n')
    print('manager.test_result \n', manager.test_result, '\n')
