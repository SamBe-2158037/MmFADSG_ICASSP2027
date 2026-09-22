# Max-Min Fairness-Aware Densest Subgraph Discovery for Exploring Segregation

Code for the paper *Max-Min Fairness-Aware Densest Subgraph Discovery for Exploring
Segregation* by Sam Beunckens, Emmanouil Kariotakis and Aritra Konar (KU Leuven).

The Max-min Fairness-Aware Densest Subgraph problem (MmFADSG) trades subgraph density
against the minimum represented proportion over any number of groups:

$$\max_{S \subseteq V} \; \rho(S) + \lambda \min_{\ell} \frac{|S \cap S_\ell|}{|S|}$$

SubSuperGreedy++ (SSG++) smooths the minimum with log-sum-exp and splits the numerator into a
supermodular part $2e(S) + \lambda\alpha Q_c(S)$ and a submodular part
$-\lambda(\mathrm{LSE} + \alpha Q_c)$. Each outer iteration modularizes the submodular part
around the current solution and solves the resulting densest supermodular subset problem with
SuperGreedy++. How much the groups in a subgraph mix is measured with the adjusted nominal
assortativity $r_{\text{adj}}$ (Karimi & Oliveira, 2023).

The proofs of Proposition 1, of the directions of $Q_c$ and of the bound on $\alpha$ are in the
[technical appendix](appendix/technical_appendix.pdf).

## Contents

```
super_greedy_set_linear.py     SSG++: log-sum-exp soft-min and its marginal gains, cross-group
                               correction Q_c, bound on alpha (default_alpha), modularizations
                               (lin_method=1: fixed marginal gains, used in the paper; 2: chain),
                               SuperGreedy++ (super_greedy_pp_lse_DC) and the outer loop
                               (ssg_pp)
utils.py                       helper functions: density (compute_density) and the adjusted nominal
                               assortativity r_adj (compute_adjusted_assortativity; Karimi & Oliveira, 2023)
init_graph.py                  loads the datasets: "cosponsorBel", "blogcatalog", "pubmed", "oklahoma"
plot_reg_path.ipynb            reproduces Tables 1-3 (SSG++ columns) and Figures 2-3, and runs
                               your own experiments
appendix/                      technical appendix: technical_appendix.pdf and its LaTeX source
datasets/                      the four datasets, see datasets/README.md
```

## Installation

```bash
pip install -r requirements.txt
```

`torch` and `torch_geometric` are only needed for PubMed, which is read with
`torch_geometric.datasets.Planetoid`. Tested with Python 3.14, NetworkX 3, NumPy 2.4,
PyTorch 2.11, PyG 2.8, Matplotlib 3.10 and seaborn 0.13.

## Usage

```python
from init_graph import init_graph
from super_greedy_set_linear import ssg_pp

G, groups = init_graph("cosponsorBel")        # run from the repository root
S, objective, groups_in_S, soft_min = ssg_pp(
    G, G, groups, lam=100, mu=2.0, alpha=None, num_passes=5, num_outer=5, lin_method=1)
```

## Reproducing the paper

Open `plot_reg_path.ipynb` from the repository root and run the imports, then any section:

| Section | What it does |
|---|---|
| Reproduce Tables 1-3 | prints the SSG++ columns of Tables 1, 2 and 3 
| Run your own experiment | one dataset at a $\lambda$, $\mu$, $t$ and $T$ of your choice 
| Regularization path | a $\lambda$ grid up to the first perfectly balanced solution 
| Figure 2 | regularization path of Bel for varying $\mu$ ($T = 10$ outer iterations) 
| Figure 3 | the Belgian network, densest subgraph and balanced SSG++ subgraph 

The experiments use $\mu = 2$, $\alpha$ at its bound, $t = 5$ SuperGreedy++ passes,
$T = 5$ outer iterations and fixed marginal gains. Table 3 uses the first perfectly balanced
$\lambda$: on the grid `linspace(0, 5000, 496)` for Bel ($\lambda \approx 90.9$) and BlogCatalog
($\lambda \approx 4353.5$), and $\lambda = 100$ and $\lambda = 1400$ for PubMed and Oklahoma.
The DDSP columns of Table 3 were computed with the implementation of Miyauchi et al. (KDD 2023)
and are not part of this repository. Figure 1 is a schematic.

