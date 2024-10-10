import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP


class HGIN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_classes,
        multi_label,
        IA_dim=0,
        n_layers=5,
        n_mlp_layers=2,
        neighbor_pooling_type="sum",
        readout_type="mean",
        drop_rate=0.5,
    ):
        super(HGIN, self).__init__()
        self.n_layers = n_layers
        self.multi_label = multi_label
        self.neighbor_pooling_type = neighbor_pooling_type
        self.readout_type = readout_type
        self.drop_rate = drop_rate
        self.mlps = nn.ModuleList([MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        self.pred_heads = nn.ModuleList([nn.Linear(in_dim, n_classes)])
        self.concat_pred_head = nn.Linear(hidden_dim * (n_layers-1), n_classes)
        self.IA_dim = IA_dim

        if IA_dim > 0:
            self.IA_head = MLP(n_mlp_layers, IA_dim, hidden_dim//2, hidden_dim//2)
            self.ori_head = MLP(n_mlp_layers, in_dim - IA_dim, hidden_dim//2, hidden_dim//2)

        for _ in range(n_layers - 2):
            self.mlps.append(MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(n_layers - 1):
            self.pred_heads.append(nn.Linear(hidden_dim, n_classes))

    def norm_row(self, H):
        N = H.shape[0]
        D_neg_1 = 1 / torch.sparse.sum(H, dim=1).to_dense()
        D_neg_1[torch.isinf(D_neg_1)] = 1
        D_neg_1 = torch.sparse_coo_tensor(torch.arange(N).repeat([2, 1]).to(H.device), D_neg_1, size=(N, N))
        return D_neg_1.mm(H)

    def forward(self, inputs):
        X, H, P, Ns, Ms = inputs
        H_T = H.transpose(0, 1).clone()

        if self.IA_dim > 0:
            IA_feat = X[..., -self.IA_dim:]
            IA_feat = F.relu(self.IA_head(IA_feat))
            X = X[..., :-self.IA_dim]
            X = F.relu(self.ori_head(X))
            X = torch.cat([X, IA_feat], dim=-1)
            X = F.relu(self.bns[0](X))

        # grpah pool
        if self.readout_type == "mean":
            P = self.norm_row(P)

        if self.neighbor_pooling_type == "mean":
            H = self.norm_row(H)
            H_T = self.norm_row(H_T)

        hidden_rep = [X]
        offset = 0 if self.IA_dim == 0 else 1
        for idx in range(offset, self.n_layers - 1):
            X = H.mm(H_T.mm(X))
            X = self.mlps[idx](X)
            X = F.relu(self.bns[idx](X))
            hidden_rep.append(X)

        out = 0
        for idx, h in enumerate(hidden_rep):
            h = P.mm(h)
            out += F.dropout(self.pred_heads[idx+offset](h), self.drop_rate, self.training)


        if self.multi_label:
            return out.sigmoid()
        else:
            return out

        
            