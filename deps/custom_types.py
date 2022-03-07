from abc import ABC
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Dict, Generic, TypeVar, Union, TypedDict

from hcve_lib.custom_types import DictAccess, ClassMapping, Printable, DataStructure
from pandas import Series, DataFrame
from sklearn.pipeline import Pipeline

T1 = TypeVar('T1')


class Estimator(ABC):
    def fit(self, X, y) -> None:
        ...

    def predict(self, X) -> Any:
        ...

    def predict_proba(self, X) -> Any:
        ...

    def set_params(self, **kwargs):
        ...


# noinspection PyUnresolvedReferences
Estimator.register(Pipeline)


class ClusteringEstimator(Estimator):
    def fit_predict(self, X, y=None) -> Any:
        ...


@dataclass(repr=False)
class ClassificationMetricsWithSTD(DataStructure):
    recall: Tuple[float, float]
    precision: Tuple[float, float]
    f1: Tuple[float, float]
    tnr: Tuple[float, float]
    fpr: Tuple[float, float]
    fnr: Tuple[float, float]
    npv: Optional[Tuple[float, float]] = None
    accuracy: Optional[Tuple[float, float]] = None
    roc_auc: Optional[Tuple[float, float]] = None
    average_precision: Optional[Tuple[float, float]] = None
    balanced_accuracy: Optional[float] = None
    brier_score: Optional[Tuple[float, float]] = None


class DataStructure(DictAccess, ClassMapping, Printable):
    ...


@dataclass
class ModelCVResult(DataStructure):
    y_train_predicts: List[Series]
    y_predicts: List[Series]
    y_train_scores: List[DataFrame]
    y_scores: List[DataFrame]
    feature_importance: List[Series]
    models: List[Any]
    elapsed: float
    payloads: List[Dict]


class ObjectiveFunctionResult(TypedDict):
    configuration: Dict
    metrics: Optional[ClassificationMetricsWithSTD]
    result: Optional[ModelCVResult]
    payload: Optional[Any]


class ObjectiveFunctionResultWithPayload(TypedDict):
    chosen: ObjectiveFunctionResult
    payload: Any


@dataclass
class ValueWithStatistics:
    mean: float
    std: Optional[float]
    ci: Optional[Tuple[float, float]]

    def format_to_list(self):
        from formatting import format_decimal
        return [
            self.format_mean(), f'±{format_decimal(self.std)}',
            *([self.format_ci()] if self.ci else [])
        ]

    def format_mean(self) -> str:
        from formatting import format_decimal
        return format_decimal(self.mean)

    def format_ci(self) -> str:
        from formatting import format_ci

        if not self.ci[0] or not self.ci[1]:
            return ""

        if self.ci:
            return format_ci(self.ci)
        else:
            raise AttributeError('CI not assigned')

    def format_short(self):
        if self.ci:
            return f'{self.format_mean()} ({self.format_ci()})'
        else:
            return f'{self.format_mean()}'

    def __str__(self):
        return " ".join(self.format_to_list())


@dataclass(repr=False)
class ClassificationMetricsWithStatistics(DataStructure):
    recall: ValueWithStatistics
    precision: ValueWithStatistics
    f1: ValueWithStatistics
    tnr: ValueWithStatistics
    fpr: ValueWithStatistics
    fnr: ValueWithStatistics
    npv: ValueWithStatistics
    accuracy: ValueWithStatistics
    roc_auc: ValueWithStatistics
    average_precision: ValueWithStatistics
    balanced_accuracy: ValueWithStatistics


@dataclass(repr=False)
class ClassificationMetrics(DataStructure):
    recall: float
    precision: float
    f1: float
    tnr: float
    fpr: float
    fnr: float
    accuracy: float
    roc_auc: float
    average_precision: float
    balanced_accuracy: float
    brier_score: float
    npv: float


@dataclass
class GenericConfusionMatrix(DataStructure, Generic[T1]):
    fn: T1
    tn: T1
    tp: T1
    fp: T1


@dataclass
class GenericConfusionMatrix(DataStructure, Generic[T1]):
    fn: T1
    tn: T1
    tp: T1
    fp: T1


ConfusionMatrix = GenericConfusionMatrix[float]
ConfusionMatrixWithStatistics = GenericConfusionMatrix[ValueWithStatistics]


class ModelResult(TypedDict):
    y_test_score: DataFrame
    y_test_predict: Series
    y_train_predict: Series
    y_train_score: DataFrame
    feature_importance: Union[Series, DataFrame]
    model: Estimator
    elapsed: float
    payload: Dict
