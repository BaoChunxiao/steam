"""
实现建模需要调用的所有函数和类
"""

import os
import logging
import pickle

import pandas as pd
from sklearn import linear_model, model_selection

from utility import timer

LOG = logging.getLogger('my')


def return_feature(method: int) -> list:
    """
    读取原始文件，根据method，返回特征集合

    :param method: 特征工程的编号
    :return: 特征集合
    """

    source_path = r'D:\study\env\steam\data\source_data\zhengqi_train.txt'
    source_data = pd.read_table(source_path, sep='\t')

    if method == 1:
        feature = list(source_data.columns)

    elif method == 2:
        feature_full = set(source_data.columns)
        drop_feature = {'V02', 'V05', 'V06', 'V09', 'V11', 'V13', 'V14', 'V17',
                        'V19', 'V20', 'V21', 'V22', 'V27', 'V35', 'V37'}
        feature = feature_full - drop_feature
        feature = sorted(list(feature))

    elif method == 3:
        feature = ['V00', 'V01', 'V08', 'V12', 'V15', 'V16', 'V18', 'V25',
                   'V29', 'V30', 'V31', 'V33', 'V34', 'V36', 'target']

    else:
        raise Exception('method输入错误')

    return feature


class FeatureManager:
    """
    特征管理器
    """

    features = {}  # 特征集合

    def add_feature(self, name: str, feature: list):
        """
        给features添加一个元素

        :param name: key
        :param feature: value
        :return: None
        """

        self.features[name] = feature


class ModelManager:
    """
    模型管理器
    """

    models = {}  # 模型集合
    param_space = {}  # 模型参数空间集合

    def add_model(self, name: str, **kwargs):
        """
        给models添加一个元素,根据name生成相应的模型

        :param name: key
        :param kwargs: 初始化模型时的参数
        :return: None
        """

        if name == 'LinearRegression':
            model = linear_model.LinearRegression(**kwargs)

        elif name == 'LassoRegression':
            model = linear_model.Lasso(**kwargs)

        elif name == 'RidgeRegression':
            model = linear_model.Ridge(**kwargs)

        else:
            raise Exception('name输入错误')

        if name in self.models.keys():
            raise Exception(f'{name} 已经存在')
        else:
            self.models[name] = model

    def update_param_space(self, name: str, param: dict):
        """
        给param_space添加或修改一个元素

        :param name: key
        :param param: value
        :return: None
        """

        if name not in self.models.keys():
            raise Exception(f'{name} 没有该模型')
        else:
            self.param_space[name] = param


class Manager(FeatureManager, ModelManager):
    """
    训练测试总流程管理器
    """

    source_train = pd.DataFrame()  # 原始训练集
    adjust_result = pd.DataFrame()  # 调参结果
    validate_result = pd.DataFrame()  # 验证结果
    test_result = pd.DataFrame()  # 测试结果

    path = r'D:\study\env\steam\data'
    pickle_path = os.path.join(path, 'steam.pickle')
    validate_path = os.path.join(path, 'validate_result.xlsx')
    test_path = os.path.join(path, 'test_result.xlsx')

    def read_source_train(self):
        """
        读原始训练集赋给source_train

        :return: None
        """

        source_path = r'D:\study\env\steam\data\source_data\zhengqi_train.txt'
        self.source_train = pd.read_table(source_path, sep='\t')

    def write_class_variable(self):
        """
        把类变量存入本地

        :return: None
        """

        object_pickle = {
            'features': self.features,
            'models': self.models,
            'param_space': self.param_space,
            'adjust_result': self.adjust_result
        }

        with open(self.pickle_path, 'wb') as f:
            pickle.dump(object_pickle, f)

        self.validate_result.to_excel(self.validate_path, index=False)
        self.test_result.to_excel(self.test_path, index=False)

    def read_class_variable(self):
        """
        读取存入本地的类变量

        :return:
        """

        self.read_source_train()

        with open(self.pickle_path, 'rb') as f:
            object_pickle = pickle.load(f)

        self.features = object_pickle['features']
        self.models = object_pickle['models']
        self.param_space = object_pickle['param_space']
        self.adjust_result = object_pickle['adjust_result']

        self.validate_result = pd.read_excel(self.validate_path)
        self.test_result = pd.read_excel(self.test_path)

    def adjust_param(self, feature_set, model_set):
        for feature_name in feature_set:
            for model_name in model_set:
                feature = self.features[feature_name]
                model = self.features[model_name]
                param = self.param_space[model_name]

                data = feature[set(feature.columns) - {'target'}]
                target = feature['target']

                clf = model_selection.GridSearchCV(model, param, cv=10)
                clf.fit(data, target)

                result = {'feature': feature_name,
                          'model': model_name,
                          'best_score': clf.best_score_,
                          'best_model': clf.best_estimator_

                          }
