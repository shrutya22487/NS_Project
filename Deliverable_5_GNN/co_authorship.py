import os

SEED = 50
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import random
import numpy as np
import torch
from collections import Counter
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.utils import from_networkx
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from networkx.algorithms.community import modularity

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

####################### HELPER FUNCTIONS #######################

def get_embeddings(data, hidden_dim=64, emb_dim=16, epochs=100, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAE(Encoder(data.num_features, hidden_dim, emb_dim)).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    return z.cpu().numpy()

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
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title(title)
    plt.xlabel("Run")
    plt.ylabel("Run")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

############################################################

################## METRICS FUNCTIONS ##################
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

#########################################################

# building encode using pytorch
class Encoder(torch.nn.Module):

    # adding 2 layers of GCNConv
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


# main function puts everything together
G_nx = nx.convert_node_labels_to_integers(nx.read_gml("../data/CA-CondMat.txt", label="id"))

data = from_networkx(G_nx)
data.x = torch.eye(G_nx.number_of_nodes(), dtype=torch.float)

emb = get_embeddings(data, epochs=200)

scores = {}
for k in range(2, 4):
    km = KMeans(n_clusters=k, random_state=SEED).fit(emb)
    scores[k] = silhouette_score(emb, km.labels_)
best_k = max(scores, key=scores.get)
print(f"Optimal number of clusters by silhouette: {best_k}")

runs = 5
km_communities = []
for i in range(runs):
    labels = KMeans(n_clusters=best_k, random_state=SEED + i).fit_predict(emb)
    comms = [[n for n, l in enumerate(labels) if l == c] for c in range(best_k)]
    km_communities.append(comms)
    if i == 0:
        plot_community_sizes(comms, f"Run {i+1} Community Sizes", f"community_sizes_run{i+1}_co.png", loglog=True)

jaccard_mat = np.zeros((runs, runs))
fsame_mat = np.zeros((runs, runs))
for i in range(runs):
    for j in range(runs):
        jaccard_mat[i, j] = jaccard_index(km_communities[i], km_communities[j])
        fsame_mat[i, j]   = compute_fsame(km_communities[i], km_communities[j])

plot_heatmap(jaccard_mat, "Jaccard Similarity Across KMeans Runs", "jaccard_confusion_co.png")
plot_heatmap(fsame_mat,   "fsame Similarity Across KMeans Runs",   "fsame_confusion_co.png")

Q_example = modularity(G_nx, km_communities[0])
print(f"Modularity of first run: {Q_example:.4f}")

