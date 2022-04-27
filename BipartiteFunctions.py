import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec
import multiprocessing

def draw_network(G, claims_nodes, dict_colors, labels, node_considered='', node_size=3, width=0.3, save=False, name="DefaultName.pdf"):
    color_map = []
    width_map = []
    for node in G:
        if node == node_considered:
            width_map.append(3)
        else:
            width_map.append(0)

        if node in claims_nodes.index:
            if labels[labels.index == node]["Fraud"][0] == 1:
                color_map.append('red')
            else:
                color_map.append('green')
        else:
            the_type = G.nodes[node]['label']
            the_color = dict_colors[the_type]
            color_map.append(the_color)
    nx.draw(G, node_color=color_map, node_size=node_size, width=width, linewidths=width_map, edgecolors='black')
    if save:
        plt.savefig(name)

def geodesic(G):
    simple_graph = nx.Graph(G)
    cycles_G = nx.cycle_basis(simple_graph)

    dict_cycle_lengths = {}
    dict_cycle_num = {}
    for cycle in cycles_G:
        for node in cycle:
            if node not in dict_cycle_lengths:
                dict_cycle_lengths[node] = []
                dict_cycle_num[node] = 0
            dict_cycle_lengths[node].append(len(cycle))
            dict_cycle_num[node] += 1

    dict_geodesic = dict((n, min(l)) for n, l in dict_cycle_lengths.items())
    df_geodesic = pd.DataFrame({'Item': [item for item in dict_geodesic],
                                'Geodesic distance': [dict_geodesic[item] for item in dict_geodesic],
                                'Number of cycles': [dict_cycle_num[item] for item in dict_cycle_num]})
    return(df_geodesic)

def Metapath2vec(G, metapaths, dimensions = 128, num_walks = 1, walk_length = 100, context_window_size = 10):
    rw = UniformRandomMetaPathWalk(G)
    walks = rw.run(
        G.nodes(), n=num_walks, length=walk_length, metapaths=metapaths
    )
    print("Number of random walks: {}".format(len(walks)))

    workers = multiprocessing.cpu_count()
    model = Word2Vec(
        walks,
        window=context_window_size,
        min_count=0,
        sg=1,
        workers=workers,
        vector_size=dimensions
    )

    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = [G.node_type(node_id) for node_id in node_ids]

    return node_ids, node_embeddings, node_targets
