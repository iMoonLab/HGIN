import dhg
import scipy
import torch
import numpy as np
import torch.nn as nn
from dhg import Graph, Hypergraph
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from skmultilearn.problem_transform import BinaryRelevance
import torch.multiprocessing as mp
import pickle
from tqdm import tqdm

# mp.set_start_method("spawn", force=True)

g2hg_func = dhg.Hypergraph.from_graph
hg2g_func = dhg.Graph.from_hypergraph_clique

def load_data(name, root, degree_as_tag, model_type, model_name=None):
    # graph dataset
    if name in ["RG_macro", "RG_sub"]:
        data_type = "graph"
        folder = "RG"
        multi_label = False
    elif name in ["MUTAG", "NCI1", "PROTEINS", "IMDBMULTI", "IMDBBINARY"]:
        data_type = "graph"
        folder = name
        multi_label = False
    elif name in ["RHG_3", "RHG_10", "RHG_table", "RHG_pyramid"]:
        data_type = "hypergraph"
        folder = "RHG"
        multi_label = False
    elif name in ["steam_player"]:
        data_type = "hypergraph"
        folder = "STEAM"
        multi_label = False
    elif name in ["IMDB_dir_genre_m"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = True
    elif name in ["IMDB_dir_form", "IMDB_dir_genre"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = False
    elif name in ["IMDB_wri_genre_m"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = True
    elif name in ["IMDB_wri_form", "IMDB_wri_genre"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = False
    elif name in ["twitter_friend"]:
        data_type = "hypergraph"
        folder = "TWITTER"
        multi_label = False
    elif name in [
        '20Random_3Regular_3Uniform_2024',
        '100Random_2Regular_3Uniform_24vertex_2024',
        '100Random_3Regular_3Uniform_15vertex_2024',
        '100Random_4Regular_2Uniform_64vertex_2024',
        '100Random_5Regular_2Uniform_40vertex_2024',
        '100Random_4Regular_3Uniform_24vertex_2024',
    ]:
        data_type = "hypergraph"
        folder = "RUHG"
        multi_label = False
    else:
        raise NotImplementedError
    if data_type == "graph" and model_type == "hypergraph":
        trans_func = g2hg_func
    elif data_type == "hypergraph" and model_type == "graph":
        trans_func = hg2g_func
    else:
        trans_func = lambda x: x

    # read data
    x_list = []
    with open(f"{root}/{data_type}/{folder}/{name}.txt", "r") as f:
        n_g = int(f.readline().strip())
        for _ in range(n_g):
            row = f.readline().strip().split()
            num_v, num_e = int(row[0]), int(row[1])
            g_lbl = [int(x) for x in row[2:]]
            v_lbl = f.readline().strip().split()
            v_lbl = [[int(x) for x in s.split('/')] for s in v_lbl]
            e_list = []
            for _ in range(num_e):
                row = f.readline().strip().split()
                e_list.append([int(x) for x in row])
            if data_type == "graph":
                d = Graph(num_v, e_list)
            else:
                d = Hypergraph(num_v, e_list)
            d = trans_func(d)
            x_list.append(
                {
                    "num_v": num_v,
                    "num_e": d.num_e,
                    "v_lbl": v_lbl,
                    "g_lbl": g_lbl,
                    "e_list": d.e[0],
                    "dhg": d,
                }
            )
    for x in x_list:
        if degree_as_tag:
            x["v_lbl"] = [int(v) for v in x["dhg"].deg_v]
        if isinstance(x["dhg"], Graph):
            x["e_lbl"] = [2] * x["num_e"]
        else:
            x["e_lbl"] = [int(e) for e in x["dhg"].deg_e]

    v_lbl_set, e_lbl_set, g_lbl_set = set(), set(), set()
    for x in x_list:
        if isinstance(x["v_lbl"][0], list):
            for v_lbl in x["v_lbl"]:
                v_lbl_set.update(v_lbl)
        else:
            v_lbl_set.update(x['v_lbl'])
        e_lbl_set.update(x["e_lbl"])
        g_lbl_set.update(x["g_lbl"])
    # re-map labels
    v_lbl_map = {x: i for i, x in enumerate(sorted(v_lbl_set))}
    e_lbl_map = {x: i for i, x in enumerate(sorted(e_lbl_set))}
    g_lbl_map = {x: i for i, x in enumerate(sorted(g_lbl_set))}
    ft_dim, n_classes = len(v_lbl_set), len(g_lbl_set)
    for x in x_list:
        x["g_lbl"] = [g_lbl_map[c] for c in x["g_lbl"]]
        if isinstance(x["v_lbl"][0], list):
            x["v_lbl"] = [tuple(sorted([v_lbl_map[c] for c in s])) for s in x["v_lbl"]]
        else:
            x["v_lbl"] = [v_lbl_map[c] for c in x["v_lbl"]]
        x["e_lbl"] = [e_lbl_map[c] for c in x["e_lbl"]]
        x["v_ft"] = np.zeros((x["num_v"], ft_dim))
        row_idx, col_idx = [], []
        for v_idx, v_lbls in enumerate(x["v_lbl"]):
            if isinstance(v_lbls, list) or isinstance(v_lbls, tuple):
                for v_lbl in v_lbls:
                    row_idx.append(v_idx)
                    col_idx.append(v_lbl)
            else:
                row_idx.append(v_idx)
                col_idx.append(v_lbls)
        x["v_ft"][row_idx, col_idx] = 1

    if model_name == "ia_hgin":
        for x in x_list:
            deg_ft = x["v_ft"]
            path_list = []
            H = x["dhg"].H.to_dense()
            adj = H.mm(H.t())
            adj_iter = torch.eye(adj.shape[0])
            for _ in range(11):
                adj_iter = adj_iter.mm(adj)
                path_list.append(adj_iter.diag().tolist())
            path_ft = np.array(path_list).T
            path_ft = np.log(path_ft)
            mu = path_ft.mean(axis=1, keepdims=True)
            sigma = path_ft.std(axis=1, keepdims=True)
            path_ft = (path_ft - mu) / (sigma + 1e-6)
            ft_mat = np.concatenate([deg_ft, path_ft], axis=1)
            x["v_ft"] = ft_mat
    try:
        ft_dim = len(x_list[0]["v_ft"][1])
    except TypeError:
        ft_dim = 1

    y_list = []
    if multi_label:
        for x in x_list:
            tmp = np.zeros(n_classes).astype(int)
            tmp[x["g_lbl"]] = 1
            y_list.append(tmp.tolist())
    else:
        y_list = [g["g_lbl"][0] for g in x_list]
    meta = {
        "multi_label": multi_label,
        "data_type": data_type,
        "ft_dim": ft_dim,
        "n_classes": len(g_lbl_set),
    }
    return x_list, y_list, meta

def load_protein_data(name, root, degree_as_tag, model_type, model_name=None):
    if name in ["EnzymeClass", "ProteinFamily", "StructuralClass_TP", "StructuralClass_CL"]:
        data_type = "hypergraph"
        folder = "PROTEIN"
        multi_label = False
    else:
        raise NotImplementedError
    if data_type == "hypergraph" and model_type == "graph":
        trans_func = hg2g_func
    else:
        trans_func = lambda x: x

    x_list = []
    with open(f"{root}/{data_type}/{folder}/{name}.pkl", "rb") as f:
        data = pickle.load(f)
    
    coords, res_labels = data["raw_data"]
    # indices = list(data["train_index"]) + list(data["test_index"]) + list(data["val_index"])
    for idx, (coord, res_label) in tqdm(enumerate(zip(coords, res_labels))):
        num_v, num_e = len(res_label), len(coord)
        g_lbl = [data["all_target"][idx]]
        v_lbl = [int(x) for x in res_label]
        e_list = data["e_list"][idx]
        d = Hypergraph(num_v, e_list)
        d = trans_func(d)
        if isinstance(d, Graph):
            e_lbl = [2] * num_e
        else:
            e_lbl = [int(e) for e in d.deg_e]
        x_list.append(
            {
                "num_v": num_v,
                "num_e": d.num_e,
                "v_lbl": v_lbl,
                "e_lbl": e_lbl,
                "g_lbl": g_lbl,
                "e_list": d.e[0],
                "dhg": d,
            }
        )

    v_lbl_set, e_lbl_set, g_lbl_set = set(), set(), set()
    for x in x_list:
        v_lbl_set.update(x["v_lbl"])
        e_lbl_set.update(x["e_lbl"])
        g_lbl_set.update(x["g_lbl"])
    e_lbl_map = {x: i for i, x in enumerate(sorted(e_lbl_set))}
    ft_dim, n_classes = 26, len(g_lbl_set)
    g_lbl_map = {x: i for i, x in enumerate(sorted(g_lbl_set))}

    for x in x_list:
        x["g_lbl"] = [g_lbl_map[c] for c in x["g_lbl"]]
        x["e_lbl"] = [e_lbl_map[c] for c in x["e_lbl"]]
        x["v_ft"] = np.zeros((x["num_v"], ft_dim))
        row_idx, col_idx = [], []
        for v_idx, v_lbl in enumerate(x["v_lbl"]):
            row_idx.append(v_idx)
            col_idx.append(v_lbl)
        x["v_ft"][row_idx, col_idx] = 1
    
    if model_name == "ia_hgin":
        for x in x_list:
            deg_ft = x["v_ft"]
            path_list = []
            H = x["dhg"].H.to_dense()
            adj = H.mm(H.t())
            adj_iter = torch.eye(adj.shape[0])
            for _ in range(10):
                adj_iter = adj_iter.mm(adj)
                path_list.append(adj_iter.diag().tolist())
            path_ft = np.array(path_list).T
            path_ft = np.log(path_ft+1)
            mu = path_ft.mean(axis=1, keepdims=True)
            sigma = path_ft.std(axis=1, keepdims=True)
            path_ft = (path_ft - mu) / (sigma + 1e-6)
            ft_mat = np.concatenate([deg_ft, path_ft], axis=1)
            x["v_ft"] = ft_mat
        ft_dim = 36 # 26(one hot) + 10(IA)
        
    y_list = [x["g_lbl"][0] for x in x_list]
    meta = {
        "multi_label": multi_label,
        "data_type": data_type,
        "ft_dim": ft_dim,
        "n_classes": n_classes,
        "train_idx": list(data["train_index"]),
        "test_idx": list(data["test_index"]),
        "val_idx": list(data["val_index"]),
    }
    return x_list, y_list, meta


def separate_data(x_list, y_list, n_fold, seed):
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    n_fold_idx = []
    for train_idx, test_idx in kf.split(x_list, y_list):
        n_fold_idx.append((train_idx, test_idx))
    return n_fold_idx


class GraphLoader:
    def __init__(self, x_list, y_list, batch_size, num_worker, shuffle, device, transform_func=None):
        if isinstance(x_list[0]["dhg"], Hypergraph):
            assert transform_func is not None
            for x in x_list:
                x["dhg"] = transform_func(x["dhg"])
        self.dataset = StructureDataset(x_list, y_list)
        self.collate_fn = create_graph_collate_fn(device)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker, collate_fn=self.collate_fn
        )

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        batches = iter(self.dataloader)
        for data, target in batches:
            data = create_batch_graph(*data)
            if len(target.shape) > 1:
                yield data, target.float()
            else:
                yield data, target.long()


class HypergraphLoader:
    def __init__(self, x_list, y_list, batch_size, num_worker, shuffle, device, transform_func=None):
        if isinstance(x_list[0]["dhg"], Graph):
            assert transform_func is not None
            for x in x_list:
                x["dhg"] = transform_func(x["dhg"])
        # x_list = [x.to(device) for x in x_list]
        self.dataset = StructureDataset(x_list, y_list)
        self.collate_fn = create_hypergraph_collate_fn(device)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_worker,
            collate_fn=self.collate_fn,
        )

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        batches = iter(self.dataloader)
        for data, target in batches:
            data = create_batch_hypergraph(*data)
            if len(target.shape) > 1:
                yield data, target.float()
            else:
                yield data, target.long()


class StructureDataset(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
        if isinstance(x_list[0]["dhg"], Hypergraph):
            self.data_type = "hypergraph"
        else:
            self.data_type = "graph"

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx]


def graph_collate_fn(batch):
    start_idx = [0]
    all_X = []
    all_Y = []
    all_N = []
    all_A_idx = []
    all_I_idx = []
    all_P_idx = []
    for idx, (x, y) in enumerate(batch):
        N = x["num_v"]
        start_idx.append(start_idx[-1] + N)
        edge_idx = x["dhg"].A._indices().clone()
        edge_idx += start_idx[idx]
        tmp = list(range(start_idx[idx], start_idx[idx + 1]))
        I_idx = torch.tensor([tmp, tmp]).long()
        P_idx = torch.tensor([[idx] * N, tmp]).long()
        all_X.append(torch.tensor(x["v_ft"]))
        all_Y.append(y)
        all_N.append(N)
        all_A_idx.append(edge_idx)
        all_I_idx.append(I_idx)
        all_P_idx.append(P_idx)
    all_X = torch.cat(all_X).float()
    all_Y = torch.tensor(all_Y)
    all_N = torch.tensor(all_N)
    all_A_idx = torch.cat(all_A_idx, dim=1)
    all_I_idx = torch.cat(all_I_idx, dim=1)
    all_P_idx = torch.cat(all_P_idx, dim=1)
    return (all_X, all_A_idx, all_I_idx, all_P_idx, all_N), all_Y


def hypergraph_collate_fn(batch):
    all_X = []
    all_Y = []
    all_N = []
    all_M = []
    all_H_idx = []
    all_P_idx = []
    bias_row, bias_col = 0, 0
    for idx, (x, y) in enumerate(batch):
        N, M = x["dhg"].num_v, x["dhg"].num_e
        H_idx = x["dhg"].H.clone()._indices()
        H_idx[0] += bias_row
        H_idx[1] += bias_col
        P_idx = torch.tensor([[idx] * N, list(range(bias_row, bias_row + N))]).long()
        bias_row += N
        bias_col += M
        all_X.append(torch.tensor(x["v_ft"]))
        all_Y.append(y)
        all_N.append(N)
        all_M.append(M)
        all_H_idx.append(H_idx)
        all_P_idx.append(P_idx)
    all_X = torch.cat(all_X).float()
    all_Y = torch.tensor(all_Y)
    all_N = torch.tensor(all_N)
    all_M = torch.tensor(all_M)
    all_H_idx = torch.cat(all_H_idx, dim=1)
    all_P_idx = torch.cat(all_P_idx, dim=1)
    return (all_X, all_H_idx, all_P_idx, all_N, all_M), all_Y

def create_graph_collate_fn(device):
    def graph_collate_fn(batch):
        start_idx = [0]
        all_X = []
        all_Y = []
        all_N = []
        all_A_idx = []
        all_I_idx = []
        all_P_idx = []
        for idx, (x, y) in enumerate(batch):
            N = x["num_v"]
            start_idx.append(start_idx[-1] + N)
            edge_idx = x["dhg"].A._indices().clone()
            edge_idx += start_idx[idx]
            tmp = list(range(start_idx[idx], start_idx[idx + 1]))
            I_idx = torch.tensor([tmp, tmp]).long()
            P_idx = torch.tensor([[idx] * N, tmp]).long()
            all_X.append(torch.tensor(x["v_ft"]))
            all_Y.append(y)
            all_N.append(N)
            all_A_idx.append(edge_idx)
            all_I_idx.append(I_idx)
            all_P_idx.append(P_idx)
        all_X = torch.cat(all_X).float().to(device)
        all_Y = torch.tensor(all_Y, device=device)
        all_N = torch.tensor(all_N, device=device)
        all_A_idx = torch.cat(all_A_idx, dim=1).to(device)
        all_I_idx = torch.cat(all_I_idx, dim=1).to(device)
        all_P_idx = torch.cat(all_P_idx, dim=1).to(device)
        return (all_X, all_A_idx, all_I_idx, all_P_idx, all_N), all_Y
    return graph_collate_fn

def create_hypergraph_collate_fn(device):
    def hypergraph_collate_fn(batch):
        all_X = []
        all_Y = []
        all_N = []
        all_M = []
        all_H_idx = []
        all_P_idx = []
        bias_row, bias_col = 0, 0
        for idx, (x, y) in enumerate(batch):
            N, M = x["dhg"].num_v, x["dhg"].num_e
            H_idx = x["dhg"].H.clone()._indices()
            H_idx[0] += bias_row
            H_idx[1] += bias_col
            P_idx = torch.tensor([[idx] * N, list(range(bias_row, bias_row + N))]).long()
            bias_row += N
            bias_col += M
            all_X.append(torch.tensor(x["v_ft"]))
            all_Y.append(y)
            all_N.append(N)
            all_M.append(M)
            all_H_idx.append(H_idx)
            all_P_idx.append(P_idx)
        all_X = torch.cat(all_X).float().to(device)
        all_Y = torch.tensor(all_Y, device=device)
        all_N = torch.tensor(all_N, device=device)
        all_M = torch.tensor(all_M, device=device)
        all_H_idx = torch.cat(all_H_idx, dim=1).to(device)
        all_P_idx = torch.cat(all_P_idx, dim=1).to(device)
        return (all_X, all_H_idx, all_P_idx, all_N, all_M), all_Y
    return hypergraph_collate_fn



def create_batch_graph(X, A_idx, I_idx, P_idx, N):
    num_x, num_v = N.shape[0], N.sum()
    A = torch.sparse_coo_tensor(A_idx, torch.ones(A_idx.shape[1]), size=(num_v, num_v), device=A_idx.device).float()
    I = torch.sparse_coo_tensor(I_idx, torch.ones(I_idx.shape[1]), size=(num_v, num_v), device=I_idx.device).float()
    P = torch.sparse_coo_tensor(P_idx, torch.ones(num_v), size=(num_x, num_v), device=P_idx.device).float()
    return X, A, I, P, N



def create_batch_hypergraph(X, H_idx, P_idx, N, M):
    num_x, num_v, num_e = N.shape[0], N.sum(), M.sum()
    H = torch.sparse_coo_tensor(H_idx, torch.ones(H_idx.shape[1], device=H_idx.device), size=(num_v, num_e), device=H_idx.device).float()
    P = torch.sparse_coo_tensor(P_idx, torch.ones(num_v, device=P_idx.device), size=(num_x, num_v), device=P_idx.device).float()
    return X, H, P, N, M

# train MLP
class MLP(nn.Module):
    def __init__(self, dim_in, n_classes, multi_label, dim_hid=64):
        super().__init__()
        self.multi_label = multi_label
        self.layers = nn.ModuleList()
        self.layer1 = nn.Sequential(
            nn.Linear(dim_in, dim_hid),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim_hid, dim_hid),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.Dropout()
        )
        self.layer2 = nn.Linear(dim_hid, n_classes)
    
    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        if self.multi_label:
            return X.sigmoid()
        else:
            return X

class MLP_Loader(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y)
        if len(self.Y.shape) > 1:
            self.Y = self.Y.float()
        else:
            self.Y = self.Y.long()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train_infer_MLP(train_X, train_Y, test_X, test_Y, n_classes, multi_label, device):
    train_dataset = MLP_Loader(train_X, train_Y)
    test_dataset = MLP_Loader(test_X, test_Y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = MLP(train_X.shape[1], n_classes, multi_label).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    if multi_label:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    best_val, best_res = 0, None
    for epoch in range(1, 40+1):
        # train
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        # test
        model.eval()
        outputs, targets = [], []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            outputs.append(output.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets)
        val, res = performance(outputs, targets, multi_label)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}--> test res: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])} \n")
        if val > best_val:
            best_val = val
            best_res = res
    return best_val, best_res

def train_infer_SVM(train_X, train_Y, test_X, test_Y, multi_label):
    if not multi_label:
        clf = SVC(kernel="precomputed")
    else:
        clf = BinaryRelevance(
            classifier=SVC(kernel="precomputed"),
            require_dense=[True, True],
        )
    clf.fit(train_X, train_Y)
    outputs = clf.predict(test_X)
    test_val, best_res = performance(outputs, test_Y, multi_label)
    return test_val, best_res

# -------------------- Metrics ----------------------------


def performance(preds: np.ndarray, targets: np.ndarray, multi_label: bool):
    if multi_label:
        if isinstance(preds, scipy.sparse.csc_matrix):
            preds = preds.todense()
        else:
            preds = (preds > 0.5).astype(int)
        # multi-label classification metric:
        # https://medium.datadriveninvestor.com/a-survey-of-evaluation-metrics-for-multilabel-classification-bb16e8cd41cd
        # acc = (preds==lbls).mean()
        # Exact Match Ratio (EMR)
        EMR = (preds == targets).all(1).mean()
        # Example-based Accuracy
        EB_acc = (np.logical_and(preds, targets).sum(1) / np.logical_or(preds, targets).sum(1)).mean()
        # Example-based Precision
        EB_pre = np.logical_and(preds, targets).sum(1) / preds.sum(1)
        EB_pre[np.isnan(EB_pre)] = 0
        EB_pre = EB_pre.mean()
        res = {"EMR": EMR, "EB_acc": EB_acc, "EB_pre": EB_pre}
        return EMR, res
    else:
        if len(preds.shape) == 2:
            preds = np.argmax(preds, axis=1)
        acc = accuracy_score(targets, preds)
        f1_micro = f1_score(targets, preds, average="micro")
        f1_macro = f1_score(targets, preds, average="macro")
        f1_weighted = f1_score(targets, preds, average="weighted")
        res = {"acc": acc, "f1_micro": f1_micro, "f1_macro": f1_macro, "f1_weighted": f1_weighted}
        return acc, res



if __name__ == "__main__":
    g_list = load_data("PTC")
    print(g_list[0])
    g_list = load_data("RHG_6_seed_0")
    print(g_list[0])
