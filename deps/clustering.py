import math
from functools import partial
# noinspection Mypy
from typing import List, Dict, TypedDict, Optional

import numpy as np
from hcve_lib.functional import star_args, statements, map_tuples, find, pipe
from pandas import DataFrame, Series
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from toolz.curried import map, sorted, get


class ClusteringInternalMetrics(TypedDict):
    si: float
    dbi: float


class ClusteringExternalMetrics(TypedDict):
    average_gini_impurity: float


class ClusteringMetrics(ClusteringInternalMetrics, ClusteringExternalMetrics):
    pass


def get_cluster_mapping_by_prevalence(
    y_pred: Series,
    y_true: Series,
) -> Dict:
    def get_1_prevalence(distribution: Series) -> int:
        try:
            prevalence_1 = (distribution[1] / distribution.sum())
        except KeyError:
            prevalence_1 = 0
        return prevalence_1

    return pipe(
        get_counts_per_cluster(y_pred, y_true),
        dict.items,
        partial(
            map_tuples, lambda index, distribution:
            (index, get_1_prevalence(distribution))),
        sorted(key=get(1)),
        enumerate,
        partial(map_tuples, lambda index, item: (item[0], index)),
        dict,
    )


class ClusteringProtocol:
    identifier: Optional[str]
    title: Optional[str]
    distance_metric: str
    parameters: Dict

    def __init__(self,
                 identifier: str = None,
                 title: str = None,
                 distance_metric: str = 'sqeuclidean',
                 parameters=None):
        self.distance_metric = distance_metric
        self.title = title
        self.identifier = identifier
        self.parameters = parameters if parameters else {}

    def get_si_score(self, X: DataFrame, y_pred: Series) -> float:
        return silhouette_score(X, y_pred, metric=self.distance_metric)

    @staticmethod
    def get_calinski_harabasz(X: DataFrame, y_pred: Series) -> float:
        return calinski_harabasz_score(X, y_pred)

    @staticmethod
    def get_db_index(X: DataFrame, y_pred: Series) -> float:
        return davies_bouldin_score(X, y_pred)

    def measure_internal_metrics(self, X: DataFrame,
                                 y_pred: Series) -> ClusteringInternalMetrics:
        try:
            si = self.get_si_score(X, y_pred)
        except ValueError:
            si = math.nan

        try:
            dbi = self.get_db_index(X, y_pred)
        except ValueError:
            dbi = math.nan

        return ClusteringInternalMetrics(
            si=si,
            dbi=dbi,
        )

    @staticmethod
    def measure_external_metrics(
            y_pred: List, y_true: List[int]) -> ClusteringExternalMetrics:
        return ClusteringExternalMetrics(
            # purity=purity_score(y_true, y_pred),
            # average_purity=average_purity(y_pred, y_true),
            average_gini_impurity=average_gini_impurity(y_pred, y_true), )

    def measure_metrics(self, X: DataFrame, y_pred: List[int],
                        y_true: List[int]) -> ClusteringMetrics:
        return ClusteringMetrics(
            **self.measure_internal_metrics(X, y_pred),
            **self.measure_external_metrics(y_pred, y_true))

    def algorithm(self, X: DataFrame, n_cluster: int) -> List[int]:
        raise NotImplemented()


class GaussianMixtureProtocol(ClusteringProtocol):
    @staticmethod
    def get_pipeline(k: int) -> Pipeline:
        # noinspection PyArgumentEqualDefault
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clustering',
             GaussianMixture(
                 n_components=k,
                 covariance_type='full',
                 n_init=50,
             )),
        ])

    def get_pipeline_with_params(self, n_clusters):
        return self.get_pipeline(k=n_clusters).set_params(**self.parameters)

    def get_bic(self, X: DataFrame, n_clusters: int) -> float:
        pipeline = self.get_pipeline_with_params(n_clusters)
        pipeline.fit(X)
        return pipeline[-1].bic(X)

    def algorithm(self, X: DataFrame, n_clusters: int) -> List[int]:
        pipeline = self.get_pipeline_with_params(n_clusters)
        y_pred = pipeline.fit_predict(X)
        return y_pred


def sort_y_proba_by_prevalence(y_proba: DataFrame,
                               y_true: Series) -> DataFrame:
    y_proba_new = y_proba.copy()

    y_pred: Series = get_y_pred_from_y_proba(y_proba)

    class_mapping = get_cluster_mapping_by_prevalence(y_pred, y_true)

    for from_class, to_class in class_mapping.items():
        y_proba_new[to_class] = y_proba[from_class]

    y_proba_new_reindexed = y_proba_new.reindex(sorted(y_proba_new.columns),
                                                axis=1)
    return y_proba_new_reindexed


def gini_impurity(points: Dict[int, int]) -> float:
    gi = 1.0
    total = sum(points)
    for label_count in points:
        gi -= (label_count / total)**2
    return gi


def get_counts_per_cluster(y_pred: Series,
                           y_true: Series) -> Dict[int, Series]:
    cluster_labels = get_cluster_identifiers(y_pred)
    true_labels = get_cluster_identifiers(y_true)
    counts = {}
    for cluster_label in cluster_labels:
        index_cluster = y_pred == cluster_label
        value_counts = y_true[index_cluster].value_counts()
        for true_label in true_labels:
            if true_label not in value_counts:
                value_counts[true_label] = 0
        counts[cluster_label] = value_counts.sort_index()
    return counts


def get_cluster_identifiers(y_pred: Series) -> List[int]:
    return list(np.unique(y_pred.tolist()))


def get_cluster_count(y_pred: List):
    return len(get_cluster_identifiers(y_pred))


def average_gini_impurity(y_pred: List[int], y_true: List[int]):
    clusters = get_counts_per_cluster(y_pred, y_true)
    rates = []
    total_count = len(y_pred)
    for cluster_index, cluster_counts in clusters.items():
        total_cluster_count = cluster_counts.sum()
        gi = gini_impurity(cluster_counts.to_dict().values())
        gi *= total_cluster_count / total_count
        rates.append(gi)
    return np.sum(rates)


def get_y_pred_from_y_proba(y_proba: DataFrame) -> Series:
    return pipe(
        y_proba.iterrows(),
        partial(
            map,
            star_args(lambda _, row: statements(
                max_value := max(list(row.values)),
                find(lambda value: value[1] == max_value, row.items())[0],
            ))),
        partial(Series, index=y_proba.index),
    )


def map_y_pred_by_prevalence(
    y_pred: Series,
    y_true: Series,
) -> Series:
    y_pred_new = y_pred.copy()
    class_mapping = get_cluster_mapping_by_prevalence(y_pred, y_true)
    for from_class, to_class in class_mapping.items():
        y_pred_new[y_pred == from_class] = to_class
    return y_pred_new
