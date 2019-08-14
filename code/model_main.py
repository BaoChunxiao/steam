"""
实现建模主流程的类
"""

import logging
import time
import json
from multiprocessing import Pool
from typing import List, Tuple, Dict, Sequence, Any

import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

from model_base import Feature, Model, Groups, Record, ReadWrite
from utility import timer, now

LOG = logging.getLogger('my')


class ModelManager(Feature, Model, Groups, Record, ReadWrite):
    """
    模型调参验证测试管理器
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

                result = result.append([model_result])

        LOG.info('调参结束')

        self.update_adjust_result(feature_list, model_list, result)

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

        :param feature_name: 特征名
        :param model_name: 模型名
        :param groups: 每组对应的训练集验证集索引
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

                model = self.search_model(feature_name, model_name)

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
                    param=json.dumps(model.get_params()),
                    update_time=now()
                )

                col = ['feature_name', 'model_name', 'group', 'train_predict',
                       'validation_predict']
                cv_predict_result = cv_predict_result.append(result[col])

        LOG.info('验证结束')

        self.update_cv_predict(feature_list, model_list, cv_predict_result)

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

                model = self.search_model(feature_name, model_name)
                model.fit(train_set, target)

                time_now = now()

                self.append_test_result(
                    feature_name=feature_name,
                    name=model_name,
                    score=None,
                    param=json.dumps(model.get_params()),
                    update_time=time_now
                )

                predict = model.predict(test_set)
                self.test_out(predict, time_now, model_name, feature_name)

        LOG.info('测试集预测结束')


class MergeManager(Feature, Model, Groups, Record, ReadWrite):
    """
    模型融合验证测试管理器
    """

    def k_fold_stacking(
            self,
            feature_name: str,
            model_list: List[str],
            groups: Dict[int, Dict[str, Sequence]],
            test_set: pd.DataFrame,
            stacking_model: str
    ) -> Tuple[Any, pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        应用于stacking的交叉验证

        :param feature_name: 特征名
        :param model_list: 模型集合
        :param groups: 每组对应的训练集验证集索引
        :param test_set: 测试集
        :param stacking_model: 应用于stacking的模型名
        :return: 训练好的stacking模型，验证集基模型预测值，验证集真值，测试集基模型预测值
        """

        feature = self.features[feature_name]

        data = self.source_train[feature]
        target = self.source_train['target']

        test_set = test_set[feature]

        validation_predict = pd.DataFrame()
        test_predict = pd.DataFrame()

        for model_name in model_list:

            model = self.search_model(feature_name, model_name)
            model_validation_predict = pd.Series()
            model_test_predict = pd.DataFrame()

            for group in groups.keys():
                train_index = groups[group]['train']
                validation_index = groups[group]['validation']

                train = data.loc[train_index]
                train_target = target[train_index]
                validation = data.loc[validation_index]

                model.fit(train, train_target)

                group_validation_predict = pd.Series(
                    model.predict(validation),
                    index=validation.index
                )

                group_test_predict = pd.Series(
                    model.predict(test_set),
                    index=test_set.index
                )

                model_validation_predict = model_validation_predict.append(
                    group_validation_predict)

                model_test_predict[group] = group_test_predict

            model_test_mean = model_test_predict.mean(axis='columns')

            validation_predict[model_name] = model_validation_predict
            test_predict[model_name] = model_test_mean

        validation_target = target[validation_predict.index]

        model = self.search_model(feature_name, stacking_model)
        model.fit(validation_predict, validation_target)

        return model, validation_predict, validation_target, test_predict

    def merge_validate(
            self,
            feature_list: List[str],
            merge_list: List[str],
            model_list: List[str],
            stacking_model: str
    ):
        """
        融合验证

        :param feature_list: 特征集合
        :param merge_list: 融合集合
        :param model_list: 融合需要的模型集合
        :param stacking_model: stacking要用的模型名
        :return:
        """

        groups = self.groups

        for feature_name in feature_list:

            feature = self.features[feature_name]
            data = self.source_train[feature]
            target = self.source_train['target']

            for merge_name in merge_list:

                LOG.info(f'开始验证 {feature_name} {merge_name}')
                result = pd.DataFrame()

                if merge_name == 'Mean':

                    cv_predict = self.cv_predict
                    choose = (cv_predict['feature_name'] == feature_name) \
                             & (cv_predict['model_name'].isin(model_list))
                    cv_predict_choose = cv_predict.loc[choose]

                    for group in groups.keys():
                        group_predict = cv_predict_choose.loc[
                            cv_predict_choose['group'] == group]

                        start_time = time.time()

                        for item in ['train', 'validation']:
                            item_predict = np.array(list(
                                group_predict[item + '_predict'].values))
                            mean_predict = item_predict.mean(axis=0)
                            true_value = target[groups[group][item]]
                            score = mean_squared_error(true_value,
                                                       mean_predict)
                            result.loc[group, item] = score

                        end_time = time.time()
                        result.loc[group, 'run_time'] = end_time - start_time

                    name = merge_name

                elif merge_name == 'Stacking':

                    for group in groups.keys():
                        inner_groups = self.stacking_groups[group].copy()

                        test_index = inner_groups['test']
                        test_set = data.loc[test_index]
                        test_target = target[test_index]

                        del inner_groups['test']

                        start_time = time.time()

                        (model,
                         validation_predict,
                         validation_target,
                         test_predict) = self.k_fold_stacking(
                            feature_name=feature_name,
                            model_list=model_list,
                            groups=inner_groups,
                            test_set=test_set,
                            stacking_model=stacking_model
                        )

                        end_time = time.time()

                        predict_validation = model.predict(validation_predict)
                        predict_test = model.predict(test_predict)

                        train_score = mean_squared_error(predict_validation,
                                                         validation_target)
                        validation_score = mean_squared_error(predict_test,
                                                              test_target)

                        result.loc[group, 'train'] = train_score
                        result.loc[group, 'validation'] = validation_score
                        result.loc[group, 'run_time'] = end_time - start_time

                    name = merge_name + '_' + stacking_model

                else:
                    raise Exception(f'merge_list参数输入错误 {merge_list}')

                self.append_validate_result(
                    feature_name=feature_name,
                    name=name,
                    score_train=result['train'].mean(),
                    score_validation=result['validation'].mean(),
                    mean_train_time=result['run_time'].mean(),
                    param=json.dumps(model_list),
                    update_time=now()
                )

        LOG.info('验证结束')

    @timer
    def merge_test(
            self,
            feature_list: List[str],
            merge_list: List[str],
            model_list: List[str],
            stacking_model: str
    ):
        """
        融合测试

        :param feature_list: 特征集合
        :param merge_list: 融合集合
        :param model_list: 融合需要的模型集合
        :param stacking_model: stacking要用的模型名
        :return:
        """

        for feature_name in feature_list:

            feature = self.features[feature_name]

            data = self.source_train[feature]
            target = self.source_train['target']

            test_set = self.test_set[feature]

            for merge_name in merge_list:

                LOG.info(f'开始测试集预测 {feature_name} {merge_name}')

                if merge_name == 'Mean':

                    result = []
                    for model_name in model_list:
                        model = self.search_model(feature_name, model_name)
                        model.fit(data, target)
                        result.append(model.predict(test_set))
                    predict = np.array(result).mean(axis=0)

                    name = merge_name

                elif merge_name == 'Stacking':

                    (model,
                     validation_predict,
                     validation_target,
                     test_predict) = self.k_fold_stacking(
                        feature_name=feature_name,
                        model_list=model_list,
                        groups=self.groups,
                        test_set=test_set,
                        stacking_model=stacking_model
                    )

                    predict = model.predict(test_predict)

                    name = merge_name + '_' + stacking_model

                else:
                    raise Exception(f'merge_list参数输入错误 {merge_list}')

                time_now = now()

                self.append_test_result(
                    feature_name=feature_name,
                    name=name,
                    score=None,
                    param=json.dumps(model_list),
                    update_time=time_now
                )

                self.test_out(predict, time_now, name, feature_name)

        LOG.info('测试集预测结束')


class Manager(ModelManager, MergeManager):
    """
    总管理器
    """

    pass
