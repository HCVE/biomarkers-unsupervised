from typing import List, Dict, Callable, Union

import networkx
from community import community_louvain
from matplotlib import pyplot
from networkx import Graph, degree, get_edge_attributes
from networkx.classes.reportviews import DegreeView
from pandas import DataFrame
from toolz import identity


def make_graph(data: DataFrame) -> Graph:
    correlation_matrix = data.corr()
    return make_graph_from_adjacency_matrix(correlation_matrix)


def make_graph_from_adjacency_matrix(correlation_matrix: DataFrame) -> Graph:
    graph = Graph()
    for index1, (feature1, row1) in enumerate(correlation_matrix.iterrows()):
        for index2, (feature2, value) in enumerate(row1.iteritems()):
            if index2 > index1:
                graph.add_edge(feature1, feature2, weight=abs(value))
    return graph


def draw_graph(
        graph: Graph, position, edge_width_scale=1, node_width_scale=1, plot_parameters: Dict = None,
        box_background: Union[bool, str] = None, modules_colors: List[str] = None,
        transform_node_dataset: Callable[[Dict], Dict] = identity,
        min_node_size: float = 0,
        node_size=None,
) -> None:
    plot_parameters = plot_parameters if plot_parameters is not None else {}
    edges, weights = zip(*get_edge_attributes(graph, 'weight').items())
    weight_for_edge_color = list(get_edge_attributes(graph, 'real_weight').values())

    if weight_for_edge_color == []:
        weight_for_edge_color = weights

    if node_size is None:
        node_size = dict(degree(graph, weight='weight'))

    nodes = node_size.keys()
    partition = community_louvain.best_partition(graph, random_state=2315)
    colors = [modules_colors[partition[feature]] if modules_colors else partition[feature] for feature in nodes]

    # noinspection PyUnresolvedReferences
    networkx.draw(
        graph,
        position,
        **{
            **dict(
                nodelist=node_size.keys(),
                edgelist=edges,
                edge_color=weight_for_edge_color,
                edge_cmap=pyplot.cm.Blues,
                edge_vmin=0,
                edge_vmax=1,
                node_color=colors,
                width=[weight * 20 * edge_width_scale for weight in weights],
                with_labels=False,
                node_size=[value * 200 * node_width_scale - min_node_size for value in
                           transform_node_dataset(node_size).values()],
            ),
            **plot_parameters,
        }
    )
    box_color = ('red' if box_background is True else box_background) if box_background not in (None, False) else None
    networkx.draw_networkx_labels(
        graph,
        pos={key: (x + 0.10, y) for key, (x, y) in position.items()},
        font_size=20,
        font_weight='bold',
        font_color='white' if box_background is not None else 'black',
        bbox=dict(
            edgecolor=box_color,
            facecolor=box_color,
        ),
        horizontalalignment='right',
        verticalalignment='bottom',
    )

    x_values, y_values = zip(*position.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.1
    pyplot.xlim(x_min - x_margin, x_max + x_margin)


def get_graph_low_threshold(adjacency_matrix: DataFrame, threshold: int) -> Graph:
    graph = Graph()
    for index1, (feature1, row1) in enumerate(adjacency_matrix.iterrows()):
        for index2, (feature2, value) in enumerate(row1.iteritems()):
            if index2 > index1:
                if abs(value) >= threshold:
                    graph.add_edge(feature1, feature2, weight=abs(value), real_weight=value)
    return graph


def get_degree_values(degrees: DegreeView) -> List[float]:
    return [value for key, value in degrees]


def get_weighted_degree(graph: Graph) -> DegreeView:
    return degree(graph, weight='weight')


def get_adjacency_power_transform(adjacency_matrix: DataFrame, beta: int) -> Graph:
    return adjacency_matrix.pow(beta)
