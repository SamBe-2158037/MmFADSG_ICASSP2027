from re import X
import networkx as nx
import numpy as np
## Utility functions for graph processing and evaluation
## Some do not see use anymore from previous iterations
def find_complement(x_vec, universe_vec):
    x_set = set(x_vec)
    universe_set = set(universe_vec)

    # Find the complement by taking the difference of the sets
    complement_elements = universe_set.difference(x_set)
    
    return complement_elements

def find_common(x_vec, y_vec):
    x_set = set(x_vec)
    y_set = set(y_vec)

    # Find the common elements by taking the intersection of the sets
    common_elements = x_set.intersection(y_set)
    
    return common_elements

def find_num_common(x_vec, y_vec):
    # Find the common elements by taking the intersection of the sets
    common_elements = find_common(x_vec, y_vec)

    # Get the number of common elements
    num_common = len(common_elements)
    
    return num_common

def find_protected_portion(x_vec, protected_vec):
    num_common = find_num_common(x_vec, protected_vec)
    return num_common / len(x_vec)

def compute_r(S, protected_nodes_all,mu=1):
    r = (1/mu)*np.log(sum([np.exp(-mu*find_num_common(S.nodes(),protected_nodes_all[i])) for i in range(0,len(protected_nodes_all))]))

    return r  

def compute_density(S, G=None, weight=None):
    if type(S) is not nx.classes.graph.Graph:
        S = G.subgraph(S)

    return S.size(weight) / S.number_of_nodes()


"""
r_adj(S): adjusted nominal assortativity (Karimi & Oliveira, Sci. Rep. 13:21053, 2023).
The mixing matrix e of S (e_ij: fraction of edge ends joining group i to group j) is rescaled by
the group shares, e*_ij = e_ij / (f_i f_j), renormalised to sum to one, and Newman's nominal
assortativity is taken on e*:  r_adj = (sum_i e*_ii - sum_i a_i^2) / (1 - sum_i a_i^2),
a_i = sum_j e*_ij. 0: groups connect to their own members as often as their total degree
predicts; > 0: homophily. Groups absent from S (f_i = 0) are dropped, vertices without a group
are ignored.
"""
def compute_adjusted_assortativity(S, protected_nodes_all):
    # node -> group
    group_of = {v: i for i, g in enumerate(protected_nodes_all) for v in g}
    L = len(protected_nodes_all)

    # mixing counts m[i][j] (i <= j) and group sizes n[i], skipping self loops and unlabelled vertices
    m = [[0.0] * L for _ in range(L)]
    for u, v in S.edges():
        if u == v or u not in group_of or v not in group_of:
            continue
        a, b = group_of[u], group_of[v]
        m[min(a, b)][max(a, b)] += 1
    n = [0] * L
    for v in S:
        if v in group_of:
            n[group_of[v]] += 1

    present = [i for i in range(L) if n[i] > 0]
    M = sum(m[i][j] for i in range(L) for j in range(i, L))
    if M == 0 or len(present) < 2:
        return float("nan")                       # undefined: no edges or a single group
    k = len(present)
    e = [[0.0] * k for _ in range(k)]
    for x, i in enumerate(present):
        e[x][x] = m[i][i] / M
        for y in range(x + 1, k):
            j = present[y]
            e[x][y] = e[y][x] = m[min(i, j)][max(i, j)] / (2 * M)

    # rescale by the group shares and renormalise
    N = sum(n)
    f = [n[i] / N for i in present]
    es = [[e[x][y] / (f[x] * f[y]) for y in range(k)] for x in range(k)]
    s = sum(sum(row) for row in es)
    es = [[x / s for x in row] for row in es]

    a = [sum(es[i][j] for j in range(k)) for i in range(k)]
    sq = sum(x * x for x in a)
    return (sum(es[i][i] for i in range(k)) - sq) / (1 - sq) if abs(1 - sq) > 1e-15 else float("nan")
