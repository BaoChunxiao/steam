"""
实现建模需要调用的所有函数和类
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

LOG = logging.getLogger('my')


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

    def add_feature(self, name: str, feature: List[str]):
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

        elif name == 'DecisionTreeRegressor':
            model = tree.DecisionTreeRegressor(**kwargs)

        elif name =='GradientBoostingRegressor':
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


class Manager(FeatureManager, ModelManager):
    """
    训练测试总流程管理器
    """

    source_train = pd.DataFrame()  # 原始训练集
    test_set = pd.DataFrame()  # 测试集
    groups = {}  # 交叉验证数据分组索引
    cv_predict = pd.DataFrame()  # 模型交叉验证预测值
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

    def create_groups(self):
        """
        返回分组对应的训练集，验证集

        :return: 每组对应的训练集验证集索引
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

    @timer
    def adjust(self, feature_list: List[str], model_list: List[str]):
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
                    model, param_space, cv=5, refit=False)
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
        adjust_result = self.update_class_result(
            self.adjust_result, result, feature_list, model_list)
        self.adjust_result = adjust_result

    def return_model(self, model_name: str, feature_name: str) -> sklearn.base:
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
            feature_name: str,
            model_name: str,
            groups: Dict[int, Dict[str, Sequence]],
            group: int,
            data: pd.DataFrame,
            target: pd.Series,
            model: sklearn.base
    ) -> Dict[str, Any]:
        """
        返回每折数据的验证结果

        :param feature_name: 特征集名
        :param model_name: 模型名
        :param groups: 每组对应的训练集验证集
        :param group: 组号
        :param data: 特征数据
        :param target: 特征数据的真值
        :param model: 训练的模型
        :return: 该组对应的验证结果
        """

        train_index = groups[group]['train']
        validation_index = groups[group]['validation']

        train = data.loc[train_index]
        train_target = target[train_index]
        validation = data.loc[validation_index]
        validation_target = target[validation_index]

        start_time = time.time()
        model.fit(train, train_target)
        end_time = time.time()

        train_predict = list(model.predict(train))
        validation_predict = list(model.predict(validation))

        train_error = mean_squared_error(train_target,
                                         train_predict)
        validation_error = mean_squared_error(validation_target,
                                              validation_predict)

        run_time = end_time - start_time

        model_result = {
            'feature_name': feature_name,
            'model_name': model_name,
            'group': group,
            'train_predict': train_predict,
            'validation_predict': validation_predict,
            'train_error': train_error,
            'validation_error': validation_error,
            'run_time': run_time,
        }
        return model_result

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

    @timer
    def model_validate(self, feature_list: List[str], model_list: List[str]):
        """
        模型验证

        :param feature_list: 特征集合
        :param model_list: 模型集合
        :return: None
        """

        groups = self.groups
        cv_predict_result = pd.DataFrame()

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
                        args=(feature_name, model_name, groups, group,
                              data, target, model)
                    )
                    result.append(group_result)
                pool.close()

                result = [group_result.get() for group_result in result]
                result = pd.DataFrame(result)

                self.append_validate_result(
                    feature_name=feature_name,
                    name=model_name,
                    score_train=result['train_error'].mean(),
                    score_validation=result['validation_error'].mean(),
                    mean_train_time=result['run_time'].mean(),
                    param=json.dumps(model.get_params())
                )

                col = ['feature_name', 'model_name', 'group', 'train_predict',
                       'validation_predict']
                cv_predict_result = cv_predict_result.append(result[col])

        LOG.info('验证结束')

        cv_predict = self.update_class_result(
            self.cv_predict, cv_predict_result, feature_list, model_list)
        self.cv_predict = cv_predict

    def merge_validate(self,
                       feature_list: List[str],
                       merge_list: List[str],
                       model_list: List[str]):
        """
        融合验证

        :param feature_list: 特征集合
        :param merge_list: 融合集合
        :param model_list: 融合需要的模型集合
        :return:
        """

        groups = self.groups
        cv_predict = self.cv_predict

        for feature_name in feature_list:

            target = self.source_train['target']
            choose = ((cv_predict['feature_name'] == feature_name)
                      & (cv_predict['model_name'].isin(model_list)))
            predict = cv_predict.loc[choose]

            for merge_name in merge_list:

                LOG.info(f'开始验证 {feature_name} {merge_name}')

                if merge_name == 'Mean':

                    result = pd.DataFrame()

                    for group in groups.keys():

                        group_predict = predict.loc[predict['group'] == group]

                        for item in ['train', 'validation']:
                            item_predict = np.array(list(
                                group_predict[item + '_predict'].values))
                            mean_predict = item_predict.mean(axis=0)
                            true_value = target[groups[group][item]]
                            score = mean_squared_error(true_value, mean_predict)
                            result.loc[group, item] = score

                    self.append_validate_result(
                        feature_name=feature_name,
                        name=merge_name,
                        score_train=result['train'].mean(),
                        score_validation=result['validation'].mean(),
                        mean_train_time=None,
                        param=json.dumps(model_list)
                    )

        LOG.info('验证结束')

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

    def test_out(self, predict, time_now, name, feature_name):
        test_predict = pd.Series(predict)
        file_name = '+'.join([time_now, name, feature_name]) + '.txt'
        path = os.path.join(self.test_predict_path, file_name)
        test_predict.to_csv(path, index=False, header=False)

    @timer
    def model_test(self, feature_list: List[str], model_list: List[str]):
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

                self.append_test_result(
                    feature_name, model_name, param, time_now)

                predict = model.predict(test_set)
                self.test_out(predict, time_now, model_name, feature_name)

        LOG.info(f'测试集预测结束')

    @timer
    def merge_test(self,
                   feature_list: List[str],
                   merge_list: List[str],
                   model_list: List[str]):
        """
        融合测试

        :param feature_list: 特征集合
        :param merge_list: 融合集合
        :param model_list: 融合需要的模型集合
        :return:
        """

        for feature_name in feature_list:

            feature = self.features[feature_name]

            train_set = self.source_train[feature]
            target = self.source_train['target']

            test_set = self.test_set[feature]

            for merge_name in merge_list:

                LOG.info(f'开始测试集预测 {feature_name} {merge_name}')

                if merge_name == 'Mean':

                    result = []
                    for model_name in model_list:
                        model = self.return_model(model_name, feature_name)
                        model.fit(train_set, target)
                        model_predict = model.predict(test_set)
                        result.append(model_predict)

                    time_now = now()
                    param = json.dumps(model_list)
                    self.append_test_result(
                        feature_name, merge_name, param, time_now)

                    predict = np.array(result).mean(axis=0)
                    self.test_out(predict, time_now, merge_name, feature_name)

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
            'cv_predict': self.cv_predict,
            'groups': self.groups
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
        self.cv_predict = object_pickle['cv_predict']
        self.groups = object_pickle['groups']

        self.validate_result = pd.read_excel(self.validate_path)
        self.test_result = pd.read_excel(self.test_path)
