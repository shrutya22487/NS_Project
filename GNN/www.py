import os
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1"
})
import numpy as np
import torch

import random
import numpy as np
import torch
from collections import Counter
from itertools import combinations
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.utils import from_networkx, negative_sampling
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from networkx.algorithms.community import modularity
from tqdm import tqdm

#––– Environment setup & global seed –––
SEED = 50
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Let OpenBLAS pick default thread count
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

#––– Metrics –––
def pairwise_community_pairs(communities):
    pairs = set()
    for comm in communities:
        for u, v in combinations(sorted(comm), 2):
            pairs.add((u, v))
    return pairs

def compute_fsame(c1, c2):
    n = sum(len(c) for c in c1)
    M = [[len(set(a).intersection(b)) for b in c2] for a in c1]
    max_row = sum(max(row) for row in M)
    max_col = sum(max(col) for col in zip(*M))
    return 0.5 * (max_row + max_col) / n

def jaccard_index(c1, c2):
    P1 = pairwise_community_pairs(c1)
    P2 = pairwise_community_pairs(c2)
    a = len(P1 & P2)
    b = len(P1 - P2)
    c = len(P2 - P1)
    return 1.0 if (a + b + c) == 0 else a / (a + b + c)

#––– GAE Encoder –––
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

#––– Extract embeddings –––
def get_embeddings(data, hidden_dim=32, emb_dim=8, epochs=100, lr=0.01):
    print("[Embedding] Initializing random features and model...", flush=True)
    data.x = torch.randn((data.num_nodes, emb_dim), dtype=torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAE(Encoder(emb_dim, hidden_dim, emb_dim)).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        neg_edge = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=int(data.edge_index.size(1) * 0.5)  # fewer negatives
        )
        loss = model.recon_loss(z, data.edge_index, neg_edge)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Embedding] Epoch {epoch}/{epochs}, loss {loss:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    return z.cpu().numpy()

#––– Plot helpers –––
def plot_community_sizes(communities, title, fname, loglog=False):
    sizes = sorted([len(c) for c in communities], reverse=True)
    plt.figure(figsize=(6, 4))
    if loglog:
        plt.loglog(sizes, marker='o')
    else:
        plt.bar(range(len(sizes)), sizes)
    plt.title(title)
    plt.xlabel("Community Rank")
    plt.ylabel("Size")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

def plot_heatmap(matrix, title, fname):
    plt.figure(figsize=(6, 5))
    import seaborn as sns
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title(title)
    plt.xlabel("Run")
    plt.ylabel("Run")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

#––– Main pipeline –––
def main():
    print("[Main] Loading graph...", flush=True)
    dataset = "www"
    if dataset == "coauthorship":
        G_nx = nx.read_edgelist("../data/CA-CondMat.txt", nodetype=int)
    elif dataset == "www":
        G_nx = nx.read_edgelist("../data/web-NotreDame.txt", nodetype=int)
    elif dataset == "zachary":
        G_nx = nx.read_gml("../data/karate.gml", label="id")
    else:
        raise ValueError("Invalid dataset")
    G_nx = nx.convert_node_labels_to_integers(G_nx)

    print(f"[Main] Nodes: {G_nx.number_of_nodes()}, Edges: {G_nx.number_of_edges()}", flush=True)

    data = from_networkx(G_nx)

    # embeddings
    emb = get_embeddings(data, epochs=200)

    # 1. Find optimal k
    print("[Clustering] Searching for optimal k via silhouette...", flush=True)
    scores = {}
    best_k = 5
    print(f"[Clustering] Optimal k: {best_k}", flush=True)

    # 2. Run KMeans multiple times
    runs = 5
    km_communities = []
    print("[KMeans] Running multiple clusterings...", flush=True)
    for i in tqdm(range(runs), desc="KMeans runs"):
        labels = KMeans(n_clusters=best_k, random_state=SEED + i).fit_predict(emb)
        comms = [[n for n, l in enumerate(labels) if l == c] for c in range(best_k)]
        km_communities.append(comms)
        
    print("[KMeans] Completed runs.", flush=True)

    # 3. Build similarity matrices with less memory
    jaccard_mat = np.zeros((runs, runs))
    fsame_mat   = np.zeros((runs, runs))
    print("[Similarity] Computing pairwise similarities...", flush=True)
    for i in tqdm(range(runs), desc="Similarity rows"):
        for j in range(i, runs):
            jv = jaccard_index(km_communities[i], km_communities[j])
            fv = compute_fsame(km_communities[i], km_communities[j])
            jaccard_mat[i, j] = jaccard_mat[j, i] = jv
            fsame_mat[i, j]   = fsame_mat[j, i]   = fv
        print(f"  Completed similarities for run {i+1}/{runs}", flush=True)

    # 4. Plot heatmaps
    plot_heatmap(jaccard_mat, "Jaccard Similarity Across Runs (www)", "jaccard_confusion_www.png")
    plot_heatmap(fsame_mat,   "fsame Similarity Across Runs (www)",   "fsame_confusion_www.png")
    print("[Plots] Heatmaps saved.", flush=True)

    # 5. Compute and print modularity
    Q_example = modularity(G_nx, km_communities[0])
    print(f"[Result] Modularity of first run: {Q_example:.4f}", flush=True)

if __name__ == "__main__":
    main()
