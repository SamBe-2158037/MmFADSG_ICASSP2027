from collections import Counter
import os
import copy
import numpy as np
import networkx as nx
from pathlib import Path
import utils
import csv

def init_graph(dataset_name):       
    if "cosponsorBel" in dataset_name:
        graph_path = "datasets/cosponsor/Bel/net_be_ch2014.gml"
        with open(graph_path, encoding='utf-8') as f:
            G_ = nx.parse_gml(f.read(), label='id')
        G_ = G_.to_undirected()

        parties = sorted({d['party'] for _, d in G_.nodes(data=True) if d.get('party')})
        party_to_int = {p: i for i, p in enumerate(parties)}
        for node, data in G_.nodes(data=True):
            nx.set_node_attributes(G_, {node: party_to_int[data['party']]}, name='value')

        G = copy.deepcopy(G_)
        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)
    elif dataset_name == "blogcatalog":
        graph_path = "datasets/BlogCatalog-dataset/data/"
        G = nx.read_edgelist(os.path.join(graph_path, "edges.csv"), nodetype=int, delimiter=',')
    elif dataset_name == "oklahoma":
        graph_path = "datasets/oklahoma/"
        G = nx.parse_gml(open(graph_path + "oklahoma97.gml", 'r').read(), label='id')
    if "cosponsor" in dataset_name:
        num_parties = len({d for _, d in G.nodes(data='value') if d is not None})
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [
            [item[0] for item in colors_list if item[1] == i]
            for i in range(num_parties)
        ]

    elif dataset_name == "blogcatalog":
        memb = set()
        with open(os.path.join(graph_path, "group-edges.csv"), encoding='utf-8') as f:
            for line in f:
                node, group = map(int, line.strip().split(","))
                memb.add((node, group))

        # 1. remove doubles, nodes that appear in more than one group
        labels_per_node = Counter(n for n, g in memb)
        single = {(n, g) for n, g in memb if labels_per_node[n] == 1}

        # 2. keep the 23 largest groups
        sizes = Counter(g for n, g in single)
        top8 = {g for g, _ in sizes.most_common(23)}

        label = {n: g for n, g in single if g in top8}

        # keep only labelled nodes, induced subgraph
        G = G.subgraph(label).copy()
        protected_nodes = [[n for n, g in label.items() if g == group] for group in top8]
        print("size of protected nodes:", np.sort([len(group) for group in protected_nodes]))
        nx.set_node_attributes(G, label, "group")
        G.remove_nodes_from(list(nx.isolates(G)))

        print(G.number_of_nodes(), G.number_of_edges())
        print(Counter(label.values()))

    elif dataset_name == "oklahoma":
            values_list = list(G.nodes(data='group'))
            protected_nodes = [[item[0] for item in values_list if item[1] == y]
                            for y in ['2005', '2006', '2007', '2008', '2009', 'other', 'NA']]
                                # class year

    elif dataset_name == "pubmed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(
            root=f'datasets/{dataset_name}',
            name=dataset_name,
        )
        data = dataset[0]
        # Build NetworkX graph
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(zip(src, dst))
        print("G is  connected?", nx.is_connected(G))
        labels = data.y.tolist()
        nx.set_node_attributes(G, {i: labels[i] for i in range(data.num_nodes)}, name='value')
        num_classes = dataset.num_classes
        protected_nodes = [
            [i for i in range(data.num_nodes) if labels[i] == c]
            for c in range(num_classes)
        ]

    return G, protected_nodes
