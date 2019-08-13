"""
实现建模主流程的类
"""

import os
import logging
import time
import json
from multiprocessing import Pool
from typing import List, Dict, Sequence, Optional, Any

import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

from model_base import (FeatureManager, ModelManager, GroupsManager,
                        RecordManager, ReadWriteManager)
from utility import timer, now

LOG = logging.getLogger('my')


class Manager(FeatureManager, ModelManager, GroupsManager, RecordManager,
              ReadWriteManager):
    """
    调参验证测试主流程管理器
    """

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
                            score = mean_squared_error(true_value,
                                                       mean_predict)
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
