
#%%
import pickle

import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, CornellDataset, TexasDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
import pickle
##############################################
# Sparse Multi-Head Attention
##############################################
#%%
class SparseMHA(nn.Module):
    def __init__(self, hidden_size=64, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h,return_attn = False):
        N = h.shape[0]

        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q = q * self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        attn = attn.softmax()  # (sparse) [N, N, nh]
        out = dglsp.bspmm(attn, v)
        out = self.out_proj(out.reshape(N, -1))
        if return_attn:
            return out, attn
        return out
    
    def get_dense_attention(self, h):
        N = h.shape[0]

        q = self.q_proj(h).reshape(N, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(N, self.num_heads, self.head_dim)

        q = q * self.scaling

        # [heads, N, N]
        scores = torch.einsum("nhd,mhd->hnm", q, k)

        # softmax over all nodes
        attn = torch.softmax(scores, dim=-1)

        return attn



##############################################
# Graph Transformer Layer
##############################################
class GTLayer(nn.Module):
    def __init__(self, hidden_size=64, num_heads=8):
        super().__init__()
        self.mha = SparseMHA(hidden_size, num_heads)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.ffn1 = nn.Linear(hidden_size, hidden_size * 2)
        self.ffn2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, A, h):
        h1 = h
        h = self.mha(A, h)
        h = self.bn1(h + h1)

        h2 = h
        h = self.ffn2(F.relu(self.ffn1(h)))
        h = self.bn2(h + h2)

        return h


##############################################
# Graph Transformer Model (Node-level)
##############################################
class GTModel(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_size=64, num_layers=6, num_heads=8, pos_enc_size=8):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_size)
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)

        self.layers = nn.ModuleList([
            GTLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])

        self.predictor = nn.Linear(hidden_size, num_classes)

    def forward(self, g, X, pos_enc):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))

        h = self.input_proj(X) + self.pos_linear(pos_enc)

        for layer in self.layers:
            h = layer(A, h)

        return self.predictor(h)  # [N, num_classes]


##############################################
# Training + Evaluation
##############################################
#%%
def train(model, g, device,train_mask, val_mask, test_mask):
    model.train()

    features = g.ndata["feat"].to(device)
    labels = g.ndata["label"].to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    best_val_acc = 0
    for epoch in range(500):
        model.train()

        logits = model(g, features, g.ndata["PE"].to(device))

        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc,logits = evaluate(model, g, device, train_mask, val_mask, test_mask)
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_output = logits
            best_accuracy = test_acc
            

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")
        with open('output_t_cornell.pickle', 'wb') as handle:
            pickle.dump(best_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(best_accuracy)

@torch.no_grad()
def evaluate(model, g, device,train_mask,val_mask,test_mask):
    model.eval()

    features = g.ndata["feat"].to(device)
    labels = g.ndata["label"].to(device)

    logits = model(g, features, g.ndata["PE"].to(device))
    pred = logits.argmax(dim=1)

    train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
    val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
    test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

    return train_acc.item(), val_acc.item(), test_acc.item(),logits


##############################################
# Main
##############################################
#%%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CornellDataset()
    g = dataset[0]

    # Add Laplacian positional encoding
    pos_enc_size = 8
    g.ndata["PE"] = dgl.lap_pe(g, k=pos_enc_size, padding=True)

    g = g.to(device)

    model = GTModel(
        in_dim=g.ndata["feat"].shape[1],
        num_classes=dataset.num_classes,
        hidden_size=64,
        num_layers=10,
        num_heads=8,
        pos_enc_size=pos_enc_size
    ).to(device)
    np.random.seed(4)
    labels = g.ndata["label"].to(device)
    labels = torch.LongTensor(labels)
    label_idx = np.where(labels >= 0)[0]
    perm = np.random.permutation(label_idx)

    n = len(perm)

    train_ratio = 0.3
    val_ratio = 0.2
    test_ratio = 0.2

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    idx_train = torch.LongTensor(perm[:n_train])
    idx_val = torch.LongTensor(perm[n_train:n_train + n_val])
    idx_test = torch.LongTensor(perm[n_train + n_val:])

    train(model, g, device, idx_train,idx_val,idx_test)
# %%
import seaborn as sns
sns.set_context("talk")
import matplotlib.pyplot as plt
import pickle
A = dglsp.spmatrix(torch.stack(g.edges()), shape=(g.num_nodes(), g.num_nodes()))

h = model.input_proj(g.ndata["feat"]) + model.pos_linear(g.ndata["PE"])

out, attn = model.layers[0].mha(A, h, return_attn=True)
attn_dense = attn.to_dense()
dense_attn_avg = attn_dense.mean(dim=2).detach().cpu().numpy()

#sort_idx = np.argsort(labels)
#A_sorted = dense_attn_avg[sort_idx, :][:, sort_idx]
with open('atten_GT_citeseer.pickle', 'wb') as handle:
   pickle.dump(dense_attn_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
