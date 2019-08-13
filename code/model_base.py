"""
实现建模基础功能的类
"""

import os
import logging
import pickle
import time
import json
from multiprocessing import Pool
from typing import List, Dict, Sequence, Optional, Any

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, tree, ensemble, svm, model_selection
from sklearn.metrics import mean_squared_error

from utility import timer, now


class PublicManager:
    """
    公共对象管理器
    """

    features = {}  # 特征集合
    models = {}  # 模型集合
    param_spaces = {}  # 模型参数空间集合

    source_train = pd.DataFrame()  # 原始训练集
    test_set = pd.DataFrame()  # 测试集

    adjust_result = pd.DataFrame()  # 调参结果记录
    validate_result = pd.DataFrame()  # 验证结果记录
    test_result = pd.DataFrame()  # 测试结果记录

    groups = {}  # 原始训练集交叉验证分组索引
    cv_predict = pd.DataFrame()  # 模型交叉验证预测值

    def search_model(self, feature_name: str, model_name: str) -> sklearn.base:
        """
        查找并返回模型

        :param feature_name: 要查找的特征名
        :param model_name: 要查找的模型名
        :return: 特征名模型名对应的具体模型
        """

        adjust_result = self.adjust_result
        is_model = (
                (adjust_result['feature_name'] == feature_name)
                & (adjust_result['model_name'] == model_name)
        )
        model_index = is_model[is_model].index[0]
        model = adjust_result.loc[model_index, 'model']
        return model


class FeatureManager(PublicManager):
    """
    特征管理器
    """

    @staticmethod
    def return_feature(method: int) -> List[str]:
        """
        读取原始文件，根据method，返回特征集合

        :param method: 特征工程的编号
        :return: 特征集合
        """

        test_path = r'D:\study\env\steam\data\source_data\zhengqi_test.txt'
        test_set = pd.read_table(test_path, sep='\t')

        if method == 1:
            feature = list(test_set.columns)

        elif method == 2:
            feature_full = set(test_set.columns)
            drop_feature = {'V02', 'V05', 'V06', 'V09', 'V11', 'V13', 'V14',
                            'V17',
                            'V19', 'V20', 'V21', 'V22', 'V27', 'V35', 'V37'}
            feature = feature_full - drop_feature
            feature = sorted(list(feature))

        elif method == 3:
            feature = ['V00', 'V01', 'V08', 'V12', 'V15', 'V16', 'V18', 'V25',
                       'V29', 'V30', 'V31', 'V33', 'V34', 'V36']

        else:
            raise Exception('method输入错误')

        return feature

    def add_feature(self, name: str, feature: List[str]):
        """
        给features添加一个元素

        :param name: key
        :param feature: value
        :return: None
        """

        self.features[name] = feature


class ModelManager(PublicManager):
    """
    模型管理器
    """

    def update_model(self, name: str, **kwargs):
        """
        给models添加或修改一个元素,根据name生成相应的模型

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

        elif name == 'DecisionTreeRegressor':
            model = tree.DecisionTreeRegressor(**kwargs)

        elif name == 'GradientBoostingRegressor':
            model = ensemble.GradientBoostingRegressor(**kwargs)

        elif name == 'RandomForestRegressor':
            model = ensemble.RandomForestRegressor(**kwargs)

        elif name == 'SVR':
            model = svm.SVR(**kwargs)

        else:
            raise Exception('name输入错误')

        self.models[name] = model

    def update_param_space(self, name: str, param: Dict[str, list]):
        """
        给param_spaces添加或修改一个元素

        :param name: key
        :param param: value
        :return: None
        """

        if name not in self.models.keys():
            raise Exception(f'{name} 没有该模型')
        else:
            self.param_spaces[name] = param


class GroupsManager(PublicManager):
    """
    原始训练集交叉验证分组管理器
    """

    def create_groups(self):
        """
        确定groups的值，即确定每组对应的训练集验证集索引

        :return: None
        """

        index = self.source_train.index
        source_train_group = pd.Series([i % 5 for i in index], index=index)
        groups = {}
        fold = [0, 1, 2, 3, 4]

        for group in fold:
            fold_temp = fold.copy()
            fold_temp.remove(group)
            train_index = list(
                source_train_group[source_train_group.isin(fold_temp)].index)
            validation_index = list(
                source_train_group[source_train_group == group].index)
            groups[group] = {
                'train': train_index,
                'validation': validation_index
            }

        self.groups = groups


class RecordManager(PublicManager):
    """
    调参验证测试，结果记录管理器
    """

    @staticmethod
    def update_class_result(class_result, result, feature_list, model_list):
        if class_result.empty:
            class_result = class_result.append(result)

        else:
            if_exist = (class_result['model_name'].isin(model_list)) \
                       & (class_result['feature_name'].isin(feature_list))

            if if_exist.sum():
                exist_index = if_exist[if_exist].index
                class_result.drop(index=exist_index, inplace=True)

            class_result = class_result.append(result)

        class_result.sort_values(['feature_name', 'model_name'], inplace=True)
        class_result.reset_index(drop=True, inplace=True)
        return class_result

    def append_validate_result(self,
                               feature_name: str,
                               name: str,
                               score_train: float,
                               score_validation: float,
                               mean_train_time: Optional[float],
                               param: str):
        """
        往self.validate_result里添加一条记录

        :param feature_name: 特征名
        :param name: 模型名或融合名
        :param score_train: 训练集平均得分
        :param score_validation: 验证集平均得分
        :param mean_train_time: 平均训练时长
        :param param: 模型或融合的参数
        :return:
        """

        validate_result = {
            'feature_name': feature_name,
            'name': name,
            'score_train': score_train,
            'score_validation': score_validation,
            'mean_train_time': mean_train_time,
            'param': param,
            'update_time': now()
        }

        self.validate_result = self.validate_result.append(
            validate_result, ignore_index=True)[validate_result.keys()]

    def append_test_result(self, feature_name, name, param, time_now):
        result = {
            'feature_name': feature_name,
            'name': name,
            'score': None,
            'param': param,
            'update_time': time_now
        }

        self.test_result = self.test_result.append(
            result, ignore_index=True)[result.keys()]


class ReadWriteManager(PublicManager):
    """
    输入输出管理器
    """

    out_path = r'D:\study\env\steam\data'  # 输出目录
    pickle_path = os.path.join(out_path, 'steam.pickle')  # pickle模块输出路径
    adjust_path = os.path.join(out_path, 'adjust_result.xlsx')  # 调参结果输出路径
    validate_path = os.path.join(out_path, 'validate_result.xlsx')  # 验证结果输出路径
    test_path = os.path.join(out_path, 'test_result.xlsx')  # 测试结果输出路径

    test_predict_path = r'D:\study\env\steam\data\predict_result'  # 测试集预测输出目录

    def read_source_train(self):
        """
        读原始训练集赋给source_train

        :return: None
        """

        source_path = r'D:\study\env\steam\data\source_data\zhengqi_train.txt'
        self.source_train = pd.read_table(source_path, sep='\t')

    def read_test_set(self):
        """
        读测试集赋给test_set

        :return: None
        """

        test_path = r'D:\study\env\steam\data\source_data\zhengqi_test.txt'
        self.test_set = pd.read_table(test_path, sep='\t')

    def test_out(self,
                 predict: Sequence[float],
                 time_now: str,
                 name: str,
                 feature_name: str):
        """
        将测试集预测结果写到文件中

        :param predict: 测试集预测值
        :param time_now: 时间
        :param name: 文件名
        :param feature_name: 特征名
        :return: None
        """

        test_predict = pd.Series(predict)
        file_name = '+'.join([time_now, name, feature_name]) + '.txt'
        path = os.path.join(self.test_predict_path, file_name)
        test_predict.to_csv(path, index=False, header=False)

    def write_class_variable(self):
        """
        把类变量存入本地

        :return: None
        """

        object_pickle = {
            'features': self.features,
            'models': self.models,
            'param_spaces': self.param_spaces,
            'adjust_result': self.adjust_result,
            'groups': self.groups,
            'cv_predict': self.cv_predict
        }

        with open(self.pickle_path, 'wb') as f:
            pickle.dump(object_pickle, f)

        self.adjust_result.to_excel(self.adjust_path, index=False)
        self.validate_result.to_excel(self.validate_path, index=False)
        self.test_result.to_excel(self.test_path, index=False)

    def read_class_variable(self):
        """
        读取存入本地的类变量

        :return:
        """

        self.read_source_train()
        self.read_test_set()

        with open(self.pickle_path, 'rb') as f:
            object_pickle = pickle.load(f)

        self.features = object_pickle['features']
        self.models = object_pickle['models']
        self.param_spaces = object_pickle['param_spaces']
        self.adjust_result = object_pickle['adjust_result']
        self.groups = object_pickle['groups']
        self.cv_predict = object_pickle['cv_predict']

        self.validate_result = pd.read_excel(self.validate_path)
        self.test_result = pd.read_excel(self.test_path)
