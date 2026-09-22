import networkx as nx
import copy
import numpy as np

import random as  rng


def compute_neg_lse(num_protected, mu=1.0):

    L = len(num_protected)
    log_terms = [-mu * n for n in num_protected]
    max_log = max(log_terms)  # log-sum-exp trick for numerical stability
    Z = sum(np.exp(t - max_log) for t in log_terms)
    return -(1.0 / mu) * (max_log + np.log(Z)) + np.log(L) / mu  


def neg_lse_marginal_add(num_protected, group_idx, mu=1.0):

    log_terms = [-mu * n for n in num_protected]
    max_log = max(log_terms)
    exps = [np.exp(t - max_log) for t in log_terms]
    Z = sum(exps)
    a = exps[group_idx]
    beta = np.exp(-mu)
    denom = Z - a * (1.0 - beta)
    if denom <= 0:
        return 0.0 
    return (1.0 / mu) * np.log(Z / denom)


def neg_lse_marginal_remove(num_protected, group_idx, mu=1.0):

    if num_protected[group_idx] <= 0:
        return 0.0
    log_terms = [-mu * n for n in num_protected]
    max_log = max(log_terms)
    exps = [np.exp(t - max_log) for t in log_terms]
    Z = sum(exps)
    a = exps[group_idx]
    e_mu = np.exp(mu)
    Z_after = Z + a * (e_mu - 1.0)
    return (1.0 / mu) * np.log(Z_after / Z)


def compute_q_cross(num_protected):
    """Q_cross = (|S|^2 - sum n_l^2) / 2"""
    total = sum(num_protected)
    sum_sq = sum(n * n for n in num_protected)
    return (total * total - sum_sq) / 2.0


def q_cross_marginal(num_protected, group_idx):
    return sum(n for i, n in enumerate(num_protected) if i != group_idx)




def default_alpha(mu, L):
    beta = np.exp(-mu)
    return np.log(1 + (1-beta)**2 *np.exp(mu)/ (4)) / mu


def compute_f_tilde(num_protected, mu=1.0, alpha=None):
    L = len(num_protected)
    if alpha is None:
        alpha = default_alpha(mu, L)
    return compute_neg_lse(num_protected, mu) - alpha * compute_q_cross(num_protected) 


def f_tilde_marginal_add(num_protected, group_idx, mu=1.0, alpha=None):
    L = len(num_protected)
    if alpha is None:
        alpha = default_alpha(mu, L)
    return (neg_lse_marginal_add(num_protected, group_idx, mu)
            - alpha * q_cross_marginal(num_protected, group_idx))


def f_tilde_marginal_remove(num_protected, group_idx, mu=1.0, alpha=None):
    L = len(num_protected)
    if alpha is None:
        alpha = default_alpha(mu, L)
    return (neg_lse_marginal_remove(num_protected, group_idx, mu)
            - alpha * q_cross_marginal(num_protected, group_idx))


def compute_linearization(G,X_nodes, all_nodes, protected_nodes_all,
                          node_to_group, mu=1.0, alpha=None, lin_method=1, rng=None):
    """
    Methods:
    1: Explicit marginal computation (O(nL))
    2: Chain ordering by marginal values at X (O(n log n))
    3: Symmetric subgradient (O(nL) but more robust in practice)
    4: Chain ordering by marginal values at X, reversed (O(n log n), worst subgradient)


    """
    L = len(protected_nodes_all)
    if alpha is None:
        alpha = default_alpha(mu, L)
    if rng is None:
        rng = np.random.default_rng()

    X_set = set(X_nodes)

    # Group counts at the linearization point X.
    n_X = [0] * L
    for v in X_set:
        n_X[node_to_group[v]] += 1

    f_tilde_X = compute_f_tilde(n_X, mu, alpha)

    h_X = {}


    if lin_method == 1:
        for v in all_nodes:
            l_v = node_to_group[v]
            if v in X_set:
                h_X[v] = f_tilde_marginal_remove(n_X, l_v, mu, alpha)
            else:
                h_X[v] = f_tilde_marginal_add(n_X, l_v, mu, alpha)

        sum_hX_over_X = sum(h_X[v] for v in X_set)
        return f_tilde_X, h_X, sum_hX_over_X


    out_nodes = [v for v in all_nodes if v not in X_set]
    subgroups = [G.subgraph(group) for group in protected_nodes_all]
    if lin_method == 2:

        subgroups = [G.subgraph(group) for group in protected_nodes_all]
        prefix_order = sorted(
            X_set,
            key=lambda v: (
                -f_tilde_marginal_remove(n_X, node_to_group[v], mu, alpha),
                v, #tie-breaker: number of neighbors in the same group, more neighbors means it will be removed later
            ),
        )
        suffix_order = sorted(
            out_nodes,
            key=lambda v: (
                f_tilde_marginal_add(n_X, node_to_group[v], mu, alpha),
                v
            ),
        )

    elif lin_method == 3:
            num_avg_chains = 1
            h_acc = dict.fromkeys(all_nodes, 0.0)
            for _ in range(num_avg_chains):
                prefix_order = list(X_set)
                rng.shuffle(prefix_order)
                suffix_order = list(out_nodes)
                rng.shuffle(suffix_order)

                n_running = [0] * L
                for v in prefix_order:          # phase 1: build up to X
                    l = node_to_group[v]
                    h_acc[v] += f_tilde_marginal_add(n_running, l, mu, alpha)
                    n_running[l] += 1
                assert n_running == n_X, f"Chain prefix mismatch: {n_running} vs {n_X}"
                for v in suffix_order:          # phase 2: extend past X
                    l = node_to_group[v]
                    h_acc[v] += f_tilde_marginal_add(n_running, l, mu, alpha)
                    n_running[l] += 1

            inv = 1.0 / num_avg_chains
            h_X = {v: h_acc[v] * inv for v in all_nodes}
            sum_hX_over_X = sum(h_X[v] for v in X_set)
            return f_tilde_X, h_X, sum_hX_over_X

    elif lin_method == 4:

        subgroups = [G.subgraph(group) for group in protected_nodes_all]
        prefix_order = sorted(
            X_set,
            key=lambda v: (
                f_tilde_marginal_remove(n_X, node_to_group[v], mu, alpha),
                v, 
            ),
        )
        suffix_order = sorted(
            out_nodes,
            key=lambda v: (
                -f_tilde_marginal_add(n_X, node_to_group[v], mu, alpha),
                v,
            ),
        )
        
    else:
        raise ValueError(f"Unknown linearization method: {lin_method}")


    n_running = [0] * L

    # Phase 1: build up to X.
    for v in prefix_order:
        l = node_to_group[v]
        h_X[v] = f_tilde_marginal_add(n_running, l, mu, alpha)
        n_running[l] += 1

    assert n_running == n_X, f"Chain prefix mismatch: {n_running} vs {n_X}"

    # Phase 2: extend past X.
    for v in suffix_order:
        l = node_to_group[v]
        h_X[v] = f_tilde_marginal_add(n_running, l, mu, alpha)
        n_running[l] += 1

    sum_hX_over_X = sum(h_X[v] for v in X_set)
    return f_tilde_X, h_X, sum_hX_over_X

# single pass of supergreedy++ with linearization of the fairness term, for a single lambda
def super_greedy_pp_lse_DC(G, S, protected_nodes_all, lam,
                              mu=1.0, alpha=None, lin_method=1, num_passes=5):

    L = len(protected_nodes_all)
    if alpha is None:
        alpha = default_alpha(mu, L)

    if G.number_of_edges() == 0:
        return set(), 0.0, [set() for _ in range(L)], 0.0

    all_nodes = set(G.nodes)

    # Extract node set from S (could be a Graph or a set)
    if hasattr(S, 'nodes'):
        S_nodes = set(S.nodes())
    else:
        S_nodes = set(S)
    S_nodes = S_nodes & all_nodes

    # Build node-to-group mapping
    node_to_group = {}
    for i, group in enumerate(protected_nodes_all):
        for node in group:
            node_to_group[node] = i

    # Frozen linearization of f_tilde around S
    _, h_X, _ = compute_linearization(G,
        S_nodes, all_nodes, protected_nodes_all, node_to_group, mu, alpha, lin_method
    )

    best_density = -np.inf
    best_subgraph = set()
    protected_nodes_densest = [set() for _ in range(L)]
    best_neg_lse = 0.0

    # Loads accumulate across passes (SuperGreedy++ property)
    loads = dict.fromkeys(G.nodes, 0.0)
    la = lam * alpha  # precompute

    for pass_idx in range(num_passes):
        remaining_nodes = set(G.nodes)
        num_edges = G.number_of_edges()
        current_degrees = dict(G.degree)

        # Live group sizes and sets
        num_protected = [0] * L
        protected_sets = [set() for _ in range(L)]
        for v in remaining_nodes:
            l_v = node_to_group[v]
            num_protected[l_v] += 1
            protected_sets[l_v].add(v)


        total_remaining = sum(num_protected)
        cross_offset = [la * (total_remaining - num_protected[l]) for l in range(L)]

        heaps = [nx.utils.BinaryHeap() for _ in range(L)]
        for node in G.nodes:
            l_v = node_to_group[node]
            base_pri = loads[node] + current_degrees[node] + lam * h_X[node]
            heaps[l_v].insert(node, base_pri)

        while remaining_nodes:
            tot_S = len(remaining_nodes)

            # Track best using TRUE objective
            neg_lse_val = compute_neg_lse(num_protected, mu)
            true_obj = 2 * num_edges + lam * neg_lse_val
            current_density = true_obj / tot_S

            if current_density > best_density:
                best_density = current_density
                best_neg_lse = neg_lse_val
                best_subgraph = set(remaining_nodes)
                protected_nodes_densest = [copy.copy(ps) for ps in protected_sets]


            best_node = None
            best_eff_pri = np.inf
            best_group = -1

            for l in range(L):
                while True:
                    try:
                        cand, base_pri = heaps[l].min()
                    except (IndexError, nx.NetworkXError):
                        break  # heap empty
                    if cand not in remaining_nodes:
                        heaps[l].pop()  # stale entry, discard
                        continue
                    eff_pri = base_pri + cross_offset[l]
                    if eff_pri < best_eff_pri:
                        best_eff_pri = eff_pri
                        best_node = cand
                        best_group = l
                    break  # found valid top of this heap

            if best_node is None:
                break

            node = best_node
            l_removed = best_group
            heaps[l_removed].pop()

            # Accumulate loads
            cross_marg_at_removal = total_remaining - num_protected[l_removed]
            loads[node] += current_degrees[node] + la * cross_marg_at_removal

            # Collect neighbors
            neighbors_in_remaining = [
                nb for nb in G.neighbors(node) if nb in remaining_nodes and nb != node
            ]

            # Update neighbor degrees and re-insert into their group heap
            for neighbor in neighbors_in_remaining:
                current_degrees[neighbor] -= 1
                num_edges -= 1
                l_n = node_to_group[neighbor]
                base_pri = loads[neighbor] + current_degrees[neighbor] + lam * h_X[neighbor]
                heaps[l_n].insert(neighbor, base_pri)  # decrease-key

            # Update group sizes and cross offsets
            num_protected[l_removed] -= 1
            protected_sets[l_removed].discard(node)
            remaining_nodes.remove(node)
            total_remaining -= 1


            for l in range(L):
                cross_offset[l] = la * (total_remaining - num_protected[l])

    return best_subgraph, best_density, protected_nodes_densest, best_neg_lse



def ssg_pp(G, S_init, protected_nodes_all, lam,
                                  mu=1.0, alpha=None,
                                  num_passes=3, num_outer=5, lin_method=1):

    L = len(protected_nodes_all)
    if alpha is None:
        alpha = default_alpha(mu, L)

    S = S_init
    best_result = None
    best_density = -np.inf
    prev_subgraph = None


    current_obj = -np.inf

    for outer_iter in range(num_outer):
        result = super_greedy_pp_lse_DC(
            G, S, protected_nodes_all, lam,
            mu=mu, alpha=alpha, lin_method=lin_method, num_passes=num_passes
        )

        current_subgraph = result[0]
        current_density = result[1]   # true objective [2e + lam*(-r)] / |S|


        if current_density > best_density:
            best_density = current_density
            best_result = result

        #  Acceptance gate 
        # Only move the linearization point if the objective did not decrease.
        if current_density >= current_obj:
            # Accept: this iterate becomes the next linearization point.
            if prev_subgraph is not None and current_subgraph == prev_subgraph:
                #print(f'Converged at outer iteration {outer_iter}')
                break
            prev_subgraph = current_subgraph
            S = G.subgraph(current_subgraph)
            current_obj = current_density
        else:
            # Reject: the step would worsen the objective. Stop.
            #print(f'Rejected non-improving step at outer iteration {outer_iter}; '
            #      f'stopping (obj {current_density:.6f} < {current_obj:.6f}).')
            break

    return best_result