"""
实现建模需要调用的所有函数和类
"""

import os
import logging
import pickle
import time
import json
from multiprocessing import Pool

import pandas as pd
import sklearn
from sklearn import linear_model, model_selection
from sklearn.metrics import mean_squared_error

from utility import timer, now

LOG = logging.getLogger('my')


def return_feature(method: int) -> list:
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
        drop_feature = {'V02', 'V05', 'V06', 'V09', 'V11', 'V13', 'V14', 'V17',
                        'V19', 'V20', 'V21', 'V22', 'V27', 'V35', 'V37'}
        feature = feature_full - drop_feature
        feature = sorted(list(feature))

    elif method == 3:
        feature = ['V00', 'V01', 'V08', 'V12', 'V15', 'V16', 'V18', 'V25',
                   'V29', 'V30', 'V31', 'V33', 'V34', 'V36']

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
    param_spaces = {}  # 模型参数空间集合

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

        else:
            raise Exception('name输入错误')

        self.models[name] = model

    def update_param_space(self, name: str, param: dict):
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


class Manager(FeatureManager, ModelManager):
    """
    训练测试总流程管理器
    """

    source_train = pd.DataFrame()  # 原始训练集
    test_set = pd.DataFrame()  # 测试集
    adjust_result = pd.DataFrame()  # 调参结果
    validate_result = pd.DataFrame()  # 验证结果
    test_result = pd.DataFrame()  # 测试结果

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

    def adjust(self, feature_list: list, model_list: list):
        """
        模型调参

        :param feature_list: 特征集合
        :param model_list: 模型集合
        :return: None
        """

        result = pd.DataFrame()

        for feature_name in feature_list:

            feature = self.features[feature_name]

            data = self.source_train[feature]
            target = self.source_train['target']

            for model_name in model_list:
                LOG.info(f'开始调参 {feature_name} {model_name}')

                model = self.models[model_name]
                param_space = self.param_spaces[model_name]

                gscv = model_selection.GridSearchCV(
                    model, param_space, cv=5, n_jobs=None, refit=False)
                start_time = time.time()
                gscv.fit(data, target)
                end_time = time.time()

                run_time = end_time - start_time
                score = gscv.best_score_
                param = gscv.best_params_
                model.set_params(**param)

                model_result = {
                    'feature_name': feature_name,
                    'model_name': model_name,
                    'model': model,
                    'score': score,
                    'run_time': run_time,
                    'param': json.dumps(param)
                }

                result = result.append(
                    model_result, ignore_index=True)[model_result.keys()]

        LOG.info('调参结束')

        result['update_time'] = now()
        adjust_result = self.adjust_result

        if adjust_result.empty:
            adjust_result = adjust_result.append(result)

        else:
            if_exist = (adjust_result['model_name'].isin(model_list)) \
                       & (adjust_result['feature_name'].isin(feature_list))

            if if_exist.sum():
                exist_index = if_exist[if_exist].index
                adjust_result.drop(index=exist_index, inplace=True)

            adjust_result = adjust_result.append(result)

        adjust_result.sort_values(['feature_name', 'model_name'], inplace=True)
        adjust_result.reset_index(drop=True, inplace=True)
        self.adjust_result = adjust_result

    def return_groups(self) -> dict:
        """
        返回分组对应的训练集，验证集

        :return: 每组对应的训练集验证集
        """

        index = self.source_train.index
        source_train_group = pd.Series([i % 5 for i in index], index=index)
        groups = {}
        fold = [0, 1, 2, 3, 4]

        for group in fold:
            fold_temp = fold.copy()
            fold_temp.remove(group)
            train_index = (
                source_train_group[source_train_group.isin(fold_temp)].index)
            validation_index = (
                source_train_group[source_train_group == group].index)
            groups[group] = {
                'train': train_index,
                'validation': validation_index
            }

        return groups

    def return_model(self, model_name: str, feature_name: str):
        """
        返回模型

        :param model_name: 要查找的模型名
        :param feature_name: 要查找的特征名
        :return: 模型名特征名对应的具体模型
        """

        adjust_result = self.adjust_result
        is_model = (
                (adjust_result['model_name'] == model_name)
                & (adjust_result['feature_name'] == feature_name)
        )
        model_index = is_model[is_model].index[0]
        model = adjust_result.loc[model_index, 'model']
        return model

    @staticmethod
    def k_fold_validation(
            groups: dict,
            group: int,
            data: pd.DataFrame,
            target: pd.Series,
            model: sklearn.base
    ) -> dict:
        """
        返回每折数据的验证结果

        :param groups: 每组对应的训练集验证集
        :param group: 组号
        :param data: 特征数据
        :param target: 特征数据的真值
        :param model: 训练的模型
        :return: 该组对应的验证结果
        """
        print('pid:', os.getpid())

        train_index = groups[group]['train']
        validation_index = groups[group]['validation']

        train = data.loc[train_index]
        train_target = target[train_index]
        validation = data.loc[validation_index]
        validation_target = target[validation_index]

        start_time = time.time()
        model.fit(train, train_target)
        end_time = time.time()

        train_predict = model.predict(train)
        validation_predict = model.predict(validation)

        train_error = mean_squared_error(train_target,
                                         train_predict)
        validation_error = mean_squared_error(validation_target,
                                              validation_predict)

        run_time = end_time - start_time

        model_result = {
            'train_error': train_error,
            'validation_error': validation_error,
            'run_time': run_time
        }
        return model_result

    def validate(self, feature_list: list, model_list: list):
        """
        模型验证

        :param feature_list: 特征集合
        :param model_list: 模型集合
        :return: None
        """

        groups = self.return_groups()

        for feature_name in feature_list:

            feature = self.features[feature_name]

            data = self.source_train[feature]
            target = self.source_train['target']

            for model_name in model_list:

                LOG.info(f'开始验证 {feature_name} {model_name}')
                model = self.return_model(model_name, feature_name)

                pool = Pool(5)
                result = []
                for group in groups.keys():
                    group_result = pool.apply_async(
                        self.k_fold_validation,
                        args=(groups, group, data, target, model)
                    )
                    result.append(group_result)
                pool.close()

                result = [group_result.get() for group_result in result]
                result = pd.DataFrame(result)

                score_train = result['train_error'].mean()
                score_validation = result['validation_error'].mean()
                mean_train_time = result['run_time'].mean()
                param = json.dumps(model.get_params())

                result = {
                    'feature_name': feature_name,
                    'name': model_name,
                    'score_train': score_train,
                    'score_validation': score_validation,
                    'mean_train_time': mean_train_time,
                    'param': param,
                    'update_time': now()
                }

                self.validate_result = self.validate_result.append(
                    result, ignore_index=True)[result.keys()]

        LOG.info('验证结束')

    def model_test(self, feature_list: list, model_list: list):
        """
        模型测试

        :param feature_list: 特征集合
        :param model_list: 模型集合
        :return: None
        """

        for feature_name in feature_list:

            feature = self.features[feature_name]

            train_set = self.source_train[feature]
            target = self.source_train['target']

            test_set = self.test_set[feature]

            for model_name in model_list:
                LOG.info(f'开始测试集预测 {feature_name} {model_name}')
                model = self.return_model(model_name, feature_name)

                model.fit(train_set, target)

                time_now = now()
                param = json.dumps(model.get_params())

                result = {
                    'feature_name': feature_name,
                    'name': model_name,
                    'score': None,
                    'param': param,
                    'update_time': time_now
                }

                self.test_result = self.test_result.append(
                    result, ignore_index=True)[result.keys()]

                test_predict = pd.Series(model.predict(test_set))
                name = '+'.join([time_now, model_name, feature_name]) + '.txt'
                path = os.path.join(self.test_predict_path, name)
                test_predict.to_csv(path, index=False, header=False)

        LOG.info(f'测试集预测结束')

    def write_class_variable(self):
        """
        把类变量存入本地

        :return: None
        """

        object_pickle = {
            'features': self.features,
            'models': self.models,
            'param_spaces': self.param_spaces,
            'adjust_result': self.adjust_result
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

        self.validate_result = pd.read_excel(self.validate_path)
        self.test_result = pd.read_excel(self.test_path)
