from typing import List, Any, Mapping, Dict

import matplotlib.pyplot as plt
# import notify2
import numpy as np
from hcve_lib.functional import flatten
from hcve_lib.utils import empty_dict
from matplotlib import rc
from pandas import Series
from sklearn.metrics import roc_curve, auc
from toolz import merge

from deps.custom_types import ModelCVResult


def plot_roc_from_results_averaged(
    y: Series,
    results: List[ModelCVResult],
    label: str = None,
    plot_kwargs: Mapping = empty_dict,
    display_random_curve: bool = True,
) -> None:
    normalized_fpr = np.linspace(0, 1, 99)

    def roc_curve_for_fold(y_score):
        fpr, tpr, thresholds = roc_curve(y.loc[y_score.index], y_score.iloc[:,
                                                                            1])
        auc_value = auc(fpr, tpr)
        normalized_tpr = np.interp(normalized_fpr, fpr, tpr)
        return normalized_tpr, auc_value

    tprs: Any
    aucs: Any
    tprs, aucs = zip(*flatten(
        [[roc_curve_for_fold(y_score) for y_score in result['y_scores']]
         for result in results]))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc: float = np.mean(aucs)
    std_auc: float = np.std(aucs, ddof=0)
    plt.plot(
        normalized_fpr, mean_tpr,
        **merge(
            dict(
                lw=1.5,
                label=f'{"ROC curve" if not label else label} (AUC=%0.3f)' %
                mean_auc,
            ),
            plot_kwargs,
        ))

    if display_random_curve:
        plt.plot([0, 1], [0, 1], color='#CCCCCC', lw=0.75, linestyle='-')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


def plot_style(grid_parameters: Dict = None, axis=None):
    rc('font', **{'family': 'Arial'})

    axis = axis or plt.gca()
    grid_parameters = grid_parameters or {}
    axis.grid(linestyle='--',
              which='major',
              color='#93939c',
              alpha=0.2,
              linewidth=1,
              **grid_parameters)
    axis.set_facecolor('white')

    for item in axis.spines.values():
        item.set_linewidth(1.4)
        item.set_edgecolor('gray')

    axis.tick_params(
        which='both',
        left=False,
        bottom=False,
        labelcolor='#314e5eff',
        labelsize=12,
    )

    axis.title.set_fontsize(15)
    axis.tick_params(axis='x', colors='black')
    axis.tick_params(axis='y', colors='black')
    axis.xaxis.label.set_fontsize(14)
    axis.xaxis.labelpad = 5
    axis.yaxis.label.set_fontsize(14)
    axis.yaxis.labelpad = 7
