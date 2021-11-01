import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

_SHORTEN_RELATION = {
    'modify': 'M',
    'located_at': 'L',
    'suggestive_of': 'S',
}

_COLOR_MAPPING = {
    'ANAT-DP': 'cadetblue',
    'OBS-DP': 'red',
    'OBS-U': 'orange',
    'OBS-DA': 'forestgreen',
}

def create_report_radgraph(entities):
    graph = nx.DiGraph()

    for entity_id, d in entities.items():
        graph.add_node(
            entity_id, tokens=d['tokens'], label=d['label'], start=d['start_ix']
        )

        for rel_type, other_id in d['relations']:
            graph.add_edge(entity_id, other_id, relation=rel_type)
    return graph

def print_id_to_tokens(graph, nodelist=None):
    nodelist = nodelist or list(graph)
    nodes_data = graph.nodes.data()

    for node in sorted(nodelist, key=int):
        info = nodes_data[node]
        print(f'{node}: {info["tokens"]} ({info["label"]})')


def _get_plot_node_colors(graph):
    nodelist = list(graph)
    nodes_data = graph.nodes.data()

    node_color = [_COLOR_MAPPING[nodes_data[node]['label']] for node in nodelist]
    return nodelist, node_color

def _get_plotable_edges(graph):
    plotable_edges = {}

    for node_1, node_2 in graph.edges:
        edge = graph[node_1][node_2]
        rel_type = edge['relation']
        rel_type = _SHORTEN_RELATION.get(rel_type, rel_type)

        plotable_edges[(node_1, node_2)] = rel_type
    return plotable_edges

def _get_plotable_labels(graph):
    nodes_data = graph.nodes.data()
    d = dict()
    for node in list(graph):
        info = nodes_data[node]
        d[node] = f'{info["tokens"]}'
    return d

def _conn_components_layout(graph, n_cols=3, n_sep=2.5, sublayout='planar'):
    sublayout_fn = nx.planar_layout if sublayout == 'planar' else nx.spring_layout

    # Calculate layouts for each subgraph
    subgraphs = [
        sublayout_fn(subgraph)
        for subgraph in nx.algorithms.components.connected_components(graph.to_undirected())
    ]

    # Sort them by number of nodes
    subgraphs = sorted(subgraphs, key=len, reverse=True)

    # Define number of rows and cols
    # n_rows = math.ceil(len(subgraphs) / n_cols)

    # New pos dict
    pos = {}
    for i, s in enumerate(subgraphs):
        row = i // n_cols
        col = i % n_cols
        x_factor = col * n_sep
        y_factor = row * n_sep
        factor = np.array([x_factor, y_factor])
        for key, value in s.items():
            pos[key] = value + factor

    return pos

def plot_radgraph(graph, figsize=(15, 8), layout=None, labels=False, **kwargs):
    assert isinstance(graph, nx.Graph), f'got {type(graph)}'

    plt.figure(figsize=figsize)
    if layout == 'planar':
        pos = nx.planar_layout(graph)
    elif layout == 'random':
        pos = nx.spring_layout(graph, k=1)
    else:
        pos = _conn_components_layout(graph, **kwargs)

    nodelist, node_color = _get_plot_node_colors(graph)
    plotable_edges = _get_plotable_edges(graph)

    nx.draw_networkx(
        graph, pos,
        with_labels=not labels, node_size=500,
        nodelist=nodelist, node_color=node_color,
    )
    nx.draw_networkx_edge_labels(
        graph, pos,
        edge_labels=plotable_edges, font_size=12,
    )
    if labels:
        nx.draw_networkx_labels(
            graph, pos,
            labels=_get_plotable_labels(graph),
        )
