from functools import partial, reduce
from statistics import mean, stdev, StatisticsError
from typing import List
from typing import Optional, Dict, Tuple, TypeVar

import numpy as np
import pandas
from hcve_lib.evaluation_functions import get_1_class_y_score, get_roc_point_by_threshold, \
    get_metrics_from_confusion_matrix, get_confusion_from_threshold
from hcve_lib.functional import flatten, statements, pass_args, mapl, pipe, unzip
from hcve_lib.stats import confidence_interval
from numpy import mean
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, \
    roc_curve, f1_score, average_precision_score, balanced_accuracy_score, \
    brier_score_loss
# noinspection Mypy
from toolz import pluck
from toolz.curried import pluck, map, valmap

from deps.custom_types import ClassificationMetrics, ValueWithStatistics, \
    ClassificationMetricsWithStatistics, ConfusionMatrix, ConfusionMatrixWithStatistics, ModelCVResult, ModelResult, \
    ObjectiveFunctionResultWithPayload
from deps.utils import get_object_attributes, object2dict

DEFAULT_THRESHOLD = 0.5
T1 = TypeVar('T1')


def compute_classification_metrics_from_results_with_statistics(
    y_true: Series,
    results: List[ModelCVResult],
    threshold: Optional[float] = None,
    target_variable: str = 'y_scores',
    ignore_warning: bool = False,
) -> ClassificationMetricsWithStatistics:
    chosen_threshold = threshold if threshold is not None else get_best_threshold_from_results(
        y_true, results)
    return pipe(
        results,
        partial(
            mapl,
            partial(
                compute_classification_metrics_from_result,
                y_true,
                threshold=chosen_threshold,
                target_variable=target_variable,
                ignore_warning=ignore_warning,
            )),
        flatten,
        list,
        compute_ci_for_metrics_collection,
    )


def compute_ci_for_metrics_collection(
        metrics: List[ClassificationMetrics]) -> Dict:
    attributes = get_object_attributes(metrics[0])
    metrics_with_ci_dict = {
        attribute: pass_args(
            confidence_interval(list(pluck(attribute, metrics))),
            lambda m, ci, std: ValueWithStatistics(m, std, ci),
        )
        for attribute in attributes
    }
    return metrics_with_ci_dict


def compute_classification_metrics_from_result(
    y: Series,
    result: ModelCVResult,
    target_variable: str = 'y_scores',
    threshold: float = DEFAULT_THRESHOLD,
    ignore_warning: bool = False,
) -> Optional[List[ClassificationMetrics]]:
    return [
        compute_classification_metrics(get_1_class_y_score(score),
                                       y,
                                       threshold=threshold,
                                       ignore_warning=ignore_warning)
        for score in result[target_variable]
    ]


def compute_classification_metrics(
        y_score,
        y_true,
        threshold: float = DEFAULT_THRESHOLD,
        ignore_warning: bool = False) -> ClassificationMetrics:
    y_score_normalized = y_score.copy()
    y_score_normalized[y_score_normalized < 0] = 0
    y_score_normalized[y_score_normalized > 1] = 1
    y_predict = y_score_normalized >= threshold
    y_true_masked = y_true.loc[y_predict.index]
    roc = roc_curve(y_true_masked, y_score_normalized)
    fpr, tpr = get_roc_point_by_threshold(threshold, *roc)
    npv = get_metrics_from_confusion_matrix(
        get_confusion_from_threshold(y_true_masked, y_score_normalized,
                                     threshold)).npv

    precision = precision_score(
        y_true_masked, y_predict,
        **({
            'zero_division': 0
        } if ignore_warning else {}))

    return ClassificationMetrics(
        recall=tpr,
        precision=precision,
        balanced_accuracy=balanced_accuracy_score(y_true_masked, y_predict),
        f1=f1_score(y_true_masked, y_predict),
        tnr=1 - fpr,
        fpr=fpr,
        fnr=1 - tpr,
        average_precision=average_precision_score(y_true_masked,
                                                  y_score_normalized),
        accuracy=accuracy_score(y_true_masked, y_predict),
        roc_auc=roc_auc_score(y_true_masked, y_score_normalized),
        npv=npv,
        brier_score=brier_score_loss(y_true_masked, y_score_normalized))


def get_best_threshold_from_results(y_true: Series,
                                    results: List[ModelCVResult]) -> float:
    fpr, tpr, thresholds = compute_threshold_averaged_roc(
        y_true, list(flatten([result['y_scores'] for result in results])))
    best_threshold, index = get_best_threshold_from_roc(tpr, fpr, thresholds)
    return best_threshold


def get_best_threshold_from_roc(
    tps: np.array,
    fps: np.array,
    thresholds: np.array,
) -> Tuple[float, int]:
    J = np.abs(tps - fps)
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh, ix


def compute_threshold_averaged_roc(
    y_true: Series,
    y_scores: List[DataFrame],
) -> Tuple[np.array, np.array, np.array]:
    def roc_curve_for_fold(y_score):
        _fpr, _tpr, thresholds = roc_curve(y_true.loc[y_score.index],
                                           get_1_class_y_score(y_score))
        return _fpr, _tpr, thresholds

    roc_curves = list([roc_curve_for_fold(y_score) for y_score in y_scores])

    all_thresholds = sorted(list(flatten([roc[2] for roc in roc_curves])),
                            reverse=True)

    def get_merged_roc_point(_roc_curves: List[Tuple[np.array, np.array,
                                                     np.array]],
                             threshold: float) -> Tuple[float, float]:
        if threshold > 1:
            threshold = 1

        merged_fpr, merged_tpr = pipe(
            _roc_curves,
            map(lambda curve: get_roc_point_by_threshold(threshold, *curve)),
            list,
            partial(np.mean, axis=0),
        )

        return merged_fpr, merged_tpr

    merged_point = [
        get_merged_roc_point(roc_curves, threshold)
        for threshold in all_thresholds
    ]
    fpr, tpr = list(unzip(merged_point))

    indexes_to_delete = []
    for index, _ in enumerate(all_thresholds):
        try:
            if fpr[index] == fpr[index + 1] or fpr[index + 1] < fpr[index]:
                indexes_to_delete.append(index)
        except IndexError:
            pass


def get_list_of_scores_from_repeated_cv_results(
        repeated_cv_results: List[ModelCVResult]) -> List[Series]:
    return list(flatten([repeat['y_scores']
                         for repeat in repeated_cv_results]))


def average_list_of_confusion_matrices(
        matrices: List[ConfusionMatrix]) -> ConfusionMatrixWithStatistics:
    return pipe(
        matrices,
        partial(map, object2dict),
        list,
        average_list_dicts,
        partial(
            valmap, lambda value: ValueWithStatistics(
                mean=value[0], std=value[1], ci=None)),
        lambda matrix: ConfusionMatrixWithStatistics(**matrix),
    )


def average_list_dicts(metrics: List[Dict]) -> Optional[Dict]:
    if len(metrics) == 0:
        return None
    output = {}
    try:
        keys = metrics[0].__dict__.keys()
    except AttributeError:
        keys = metrics[0].keys()

    for key in keys:
        values = list(
            map(
                lambda item: getattr(item, key)
                if hasattr(item, key) else item[key], metrics))

        mean_value = mean(values)

        try:
            stdev_value = stdev(values)
        except StatisticsError:
            stdev_value = 0

        output[key] = (mean_value, stdev_value)
    return output


def join_repeats_and_folds_cv_results(
        results: List[ModelCVResult]) -> ModelResult:
    return ModelResult(**pipe(
        results,
        join_repeats_cv_results,
        join_folds_cv_result,
    ))


def join_folds_cv_result(result: ModelCVResult) -> ModelResult:
    try:
        feature_importance = result['feature_importance'][0]
    except (KeyError, ValueError):
        feature_importance = None

    return ModelResult(
        feature_importance=get_feature_importance_from_cv_result(result)
        if feature_importance is not None else None,
        y_test_score=pandas.concat(result['y_scores']).sort_index(),
        y_test_predict=pandas.concat(result['y_predicts']).sort_index(),
        y_train_predict=pandas.concat(result['y_train_predicts']).sort_index(),
        y_train_score=pandas.concat(result['y_train_scores']).sort_index(),
        models=list(flatten(result['models'])),
        elapsed=result['elapsed'],
    )


def get_feature_importance_from_cv_results(
        cv_results: List[ModelCVResult]) -> DataFrame:
    return join_repeats_and_folds_cv_results(cv_results)['feature_importance']


def join_repeats_cv_results(results: List[ModelCVResult]) -> ModelCVResult:
    return reduce(
        lambda result1, result2: ModelCVResult(
            y_train_predicts=
            [*result1['y_train_predicts'], *result2['y_train_predicts']],
            y_predicts=[*result1['y_predicts'], *result2['y_predicts']],
            y_train_scores=
            [*result1['y_train_scores'], *result2['y_train_scores']],
            y_scores=[*result1['y_scores'], *result2['y_scores']],
            feature_importance=
            [*result1['feature_importance'], *result2['feature_importance']],
            models=[*result1['models'], *result2['models']],
            elapsed=result1['elapsed'] + result2['elapsed'],
        ),
        results,
    )


def get_feature_importance_from_cv_result(result: ModelCVResult) -> DataFrame:
    return statements(
        feature_importance_vector := pandas.concat(
            result['feature_importance'],
            axis=1,
        ).transpose(),
        DataFrame({
            'mean': feature_importance_vector.mean(),
            'std': feature_importance_vector.std(),
        }).sort_values(by='mean', ascending=False, inplace=False))


def get_cv_results_from_simple_cv_evaluation(
    simple_cv_result: List[ObjectiveFunctionResultWithPayload]
) -> List[ModelCVResult]:
    return mapl(lambda item: item['chosen']['result'], simple_cv_result)
