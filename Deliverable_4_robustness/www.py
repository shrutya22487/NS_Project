import random
import time
from collections import defaultdict, Counter
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
################# HELPER FUNCTIONS #####################

def plot_quality_vs_noise(robust_res):
    plt.figure(figsize=(6, 4))
    for mode in sorted({mode for mode, *_ in robust_res}):
        data = [(p, Q) for m, p, Q, *_ in robust_res if m == mode]
        ps, Qs = zip(*data)
        label = "Random removal" if mode=="remove" else "Targeted removal"
        plt.plot(ps, Qs, marker='o', label=label)
    plt.xlabel("Noise fraction p")
    plt.ylabel("Modularity Q")
    plt.title("Modularity vs. Noise Removal Type")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"modularity_vs_noise_{mode}_www.png")

def plot_stability_vs_noise(robust_res):
    plt.figure(figsize=(6, 4))
    for mode in sorted({mode for mode, *_ in robust_res}):
        data = [(p, j) for m, p, _, _, j, *_ in robust_res if m == mode]
        ps, js = zip(*data)
        label = "Random removal" if mode=="remove" else "Targeted removal"
        plt.plot(ps, js, marker='x', label=label)
    plt.xlabel("Noise fraction p")
    plt.ylabel("Jaccard Index")
    plt.title("Stability vs. Noise Removal Type")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"stability_vs_noise_{mode}_www.png")

   ############################################### 
# just builds the adjacenecy list 
def build_adjacency_list(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

# original LPA algorithm, repeats till max_iter to average out results and returns the communities
def label_propagation(adj_list, max_iter=1000, return_iters=False):
    labels = {node: node for node in adj_list}
    for it in range(1, max_iter + 1):
        nodes = list(adj_list.keys())
        random.shuffle(nodes)
        changed = False
        for node in nodes:
            neigh_labels = [labels[n] for n in adj_list[node]]
            if not neigh_labels:
                continue
            freq = Counter(neigh_labels)
            max_f = max(freq.values())
            best = [lbl for lbl, c in freq.items() if c == max_f]
            new = random.choice(best)
            if labels[node] != new:
                labels[node] = new
                changed = True
        if not changed:
            break

    comm = defaultdict(list)
    for n, lbl in labels.items():
        comm[lbl].append(n)
    communities = list(comm.values())

    if return_iters:
        return communities, it
    return communities

# to calculate jaccard index 
def jaccard_index(c1, c2):
    def pairs(comms):
        P = set()
        for c in comms:
            for u, v in combinations(sorted(c), 2):
                P.add((u, v))
        return P
    P1, P2 = pairs(c1), pairs(c2)
    a = len(P1 & P2)
    b = len(P1 - P2)
    c = len(P2 - P1)
    return 1.0 if a + b + c == 0 else a / (a + b + c)
# to calculate fsame
def compute_fsame(c1, c2):
    n = sum(len(c) for c in c1)
    M = [[len(set(a) & set(b)) for b in c2] for a in c1]
    max_row = sum(max(row) for row in M)
    max_col = sum(max(col) for col in zip(*M))
    return 0.5 * (max_row + max_col) / n

# will remove edges from the graph which can be random or targeted
def perturb_graph(G, p, mode="remove", baseline_comms=None):
    Gp = G.copy()
    E = list(Gp.edges())
    m = len(E)
    k = int(p * m)

    if mode == "remove":
        to_remove = random.sample(E, k)
        Gp.remove_edges_from(to_remove)

    elif mode == "targeted_remove":
        if baseline_comms is None:
            raise ValueError("baseline_comms required for targeted removal")
        node2comm = {}
        for idx, comm in enumerate(baseline_comms):
            for node in comm:
                node2comm[node] = idx
        inter_edges = [(u, v) for u, v in E
                       if node2comm.get(u) != node2comm.get(v)]
        to_remove = random.sample(inter_edges, min(k, len(inter_edges)))
        Gp.remove_edges_from(to_remove)

    else:
        raise ValueError(f"Unknown perturbation mode '{mode}'")

    return Gp

# main method to run LPA defined above
def run_LPA(G, max_iter):
    edges = list(G.edges())
    adj = build_adjacency_list(edges)
    for node in G.nodes():
        adj.setdefault(node, [])
    start = time.time()
    communities, iters = label_propagation(adj, max_iter, return_iters=True)
    runtime = time.time() - start
    Q = nx.algorithms.community.quality.modularity(G, communities)
    return communities, Q, iters, runtime

# this method runs the robustness experiment
def robustness_experiment(G, noise_levels, repeats, max_iter):
    baseline_comms, _, _, _ = run_LPA(G, max_iter)
    results = []

    for mode in ["remove", "targeted_remove"]:
        for p in tqdm(noise_levels, desc=f"Noise {mode}"):
            Qs, Js, Fs = [], [], []
            for _ in range(repeats):
                Gp = perturb_graph(G, p, mode, baseline_comms=baseline_comms)
                comms, Q, _, _ = run_LPA(Gp, max_iter)
                Qs.append(Q)
                Js.append(jaccard_index(baseline_comms, comms))
                Fs.append(compute_fsame(baseline_comms, comms))
            results.append((
                mode,
                p,
                np.mean(Qs), np.std(Qs),
                np.mean(Js), np.std(Js),
                np.mean(Fs), np.std(Fs)
            ))
    return results

edges = []
with open("../data/web-NotreDame.txt", 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        u, v = map(int, parts[:2])
        edges.append((u, v))

G = nx.Graph().add_edges_from(edges)

noise_levels = [0.01, 0.05, 0.10, 0.20]

result = robustness_experiment(G, noise_levels, 2, 10)
plot_quality_vs_noise(result)
plot_stability_vs_noise(result)
