import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Define DNA Sequences
sequences = {
    "Seq1": "ATGCTAGCTAGCTAGCTAGCTAGC",
    "Seq2": "ATGCTAGCTAGCTAGTTAGCTAGC",
    "Seq3": "ATGCGAGCTAGCTAGTTAGCTAGC",
    "Seq4": "ATGCTAGCTAGCTAGCTGGCTAGC"
}


# Step 2: Compute Pairwise Distance Using Needleman-Wunsch Alignment
def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1))

    # Initialize gap penalties
    for i in range(n + 1):
        dp[i][0] = i * gap
    for j in range(m + 1):
        dp[0][j] = j * gap

    # Fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_mismatch = dp[i - 1][j - 1] + (match if seq1[i - 1] == seq2[j - 1] else mismatch)
            delete = dp[i - 1][j] + gap
            insert = dp[i][j - 1] + gap
            dp[i][j] = max(match_mismatch, delete, insert)

    return dp[n][m]  # Return final alignment score


# Compute pairwise distances
labels = list(sequences.keys())
num_sequences = len(labels)
distance_matrix = np.zeros((num_sequences, num_sequences))

for i, j in itertools.combinations(range(num_sequences), 2):
    seq1, seq2 = sequences[labels[i]], sequences[labels[j]]
    distance = -needleman_wunsch(seq1, seq2)  # Convert similarity to distance
    distance_matrix[i, j] = distance_matrix[j, i] = distance


# Step 3: Apply Neighbor-Joining Algorithm
def neighbor_joining(dist_matrix, labels):
    tree = []
    while len(labels) > 2:
        n = len(dist_matrix)
        q_matrix = np.zeros((n, n))

        # Compute Q-matrix
        total_dist = np.sum(dist_matrix, axis=1)
        for i in range(n):
            for j in range(n):
                if i != j:
                    q_matrix[i][j] = (n - 2) * dist_matrix[i][j] - total_dist[i] - total_dist[j]

        # Find the pair with the minimum Q-value
        i, j = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)

        # Create a new node
        new_label = f"({labels[i]}-{labels[j]})"
        tree.append((labels[i], new_label))
        tree.append((labels[j], new_label))

        # Update distance matrix
        new_dist = [(dist_matrix[i, k] + dist_matrix[j, k]) / 2 for k in range(n) if k != i and k != j]
        new_matrix = np.zeros((n - 1, n - 1))

        idx = 0
        for k in range(n):
            if k in [i, j]:
                continue
            new_matrix[idx, :-1] = new_dist[idx]
            new_matrix[:-1, idx] = new_dist[idx]
            idx += 1

        labels = [labels[k] for k in range(n) if k != i and k != j] + [new_label]
        dist_matrix = new_matrix

    tree.append((labels[0], labels[1]))  # Final pair
    return tree


phylo_tree = neighbor_joining(distance_matrix, labels)

# Step 4: Visualize Phylogenetic Tree
G = nx.Graph()
for parent, child in phylo_tree:
    G.add_edge(parent, child)

plt.figure(figsize=(8, 5))
pos = nx.spring_layout(G)  # Auto-layout
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, edge_color="gray")
plt.title("Phylogenetic Tree (Neighbor-Joining)")
plt.show()
