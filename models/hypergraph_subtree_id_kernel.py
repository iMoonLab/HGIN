from itertools import combinations
from collections import defaultdict

import torch
import numpy as np
from dhg import Hypergraph, Graph, DiGraph

# Hypergraph Kernel from "Feng et al. 2023"
class HypergraphSubtreeIDKernel:
    def __init__(self, way="ve", n_iter=3, degree_as_label=True, normalize=True):
        assert way in ["v", "e", "ve"]
        self.way = way
        self.n_iter = n_iter
        self.normalize = normalize
        self.degree_as_label = degree_as_label
        self._subtree_map = {}
        self.n_length = n_iter

    def remap_v(self, hg_list, cnt, drop=False):
        for hg_idx, hg in enumerate(hg_list):
            for v_idx in range(hg["num_v"]):
                # 获得每个节点的当前的string
                cur_lbl = hg["v_lbl"][v_idx]
                cur_lbl = "v" + str(cur_lbl)
                # 如果没有当前label的映射，则将该label放进映射dict，对应的值为当前映射函数key的数量（保证唯一性）
                if cur_lbl not in self._subtree_map:
                    if drop:
                        hg["v_lbl"][v_idx] = -1
                        continue
                    else:
                        self._subtree_map[cur_lbl] = len(self._subtree_map)
                # relabel当前节点
                hg["v_lbl"][v_idx] = self._subtree_map[cur_lbl]
                # 映射后，我们更新对应超图在更新的label（对应该轮迭代的子树结构）位置的计数
                cnt[hg_idx][self._subtree_map[cur_lbl]] += 1
        return hg_list, cnt

    def remap_e(self, hg_list, cnt, drop=False):
        for hg_idx, hg in enumerate(hg_list):
            for e_idx in range(hg["dhg"].num_e):
                # cur_lbl 指的是汇聚了邻居信息后的 string
                cur_lbl = hg["e_lbl"][e_idx]
                cur_lbl = "e" + str(cur_lbl)
                if cur_lbl not in self._subtree_map:
                    if drop:
                        hg["e_lbl"][e_idx] = -1
                        continue
                    else:
                        # 对应于压缩过程，将string变为new label
                        self._subtree_map[cur_lbl] = len(self._subtree_map)
                hg["e_lbl"][e_idx] = self._subtree_map[cur_lbl]
                cnt[hg_idx][self._subtree_map[cur_lbl]] += 1
        return hg_list, cnt

    def cnt2mat(self, raw_cnt):
        # filter count
        cnt = []
        if self.way == "v":
            # 取出节点的所有map，将每一轮节点的label放入集合
            valid_id_set = set([v for k, v in self._subtree_map.items() if k.startswith("v")])
            # 将所有节点的label作为key，按顺序依次标号（0，1，2 ...）作为value，方便后续形成tensor的索引
            id_map = {k: v for v, k in enumerate(sorted(valid_id_set))}
            # 抽出所有v对应的key值, 将它们的cnt重新构成matrix；
            # raw_cnt 对应 每个hg具有的映射（key为label，value为计数）构成的list
            for c in raw_cnt:
                cnt.append({id_map[k]: v for k, v in c.items() if k in valid_id_set})
        elif self.way == "e":
            valid_id_set = set([v for k, v in self._subtree_map.items() if k.startswith("e")])
            id_map = {k: v for v, k in enumerate(sorted(valid_id_set))}
            for c in raw_cnt:
                cnt.append({id_map[k]: v for k, v in c.items() if k in valid_id_set})
        else:
            cnt = raw_cnt
        # count
        row_idx, col_idx, data = [], [], []
        for idx, g in enumerate(cnt):
            for lbl, c in g.items():
                row_idx.append(idx)
                col_idx.append(lbl)
                data.append(c)
        return (
            torch.sparse_coo_tensor(
                torch.tensor([row_idx, col_idx]), torch.tensor(data), size=(len(cnt), len(self._subtree_map))
            )
            .coalesce()
            .float()
        )

    def fit_transform(self, hg_list, hg_te_list=[]):
        self.adj_list = [hg["dhg"].H.mm(hg["dhg"].H.t()).to_dense() for hg in hg_list]
        self.te_adj_list = [hg["dhg"].H.mm(hg["dhg"].H.t()).to_dense() for hg in hg_te_list]
        self.paths_set_list = []
        self.path_ft = []

        for idx, hg in enumerate(hg_list):
            hg["paths"] = []
            adj_iter = torch.eye(self.adj_list[idx].shape[0])
            for _ in range(self.n_length):
                adj_iter = adj_iter.mm(self.adj_list[idx])
                hg["paths"].append(adj_iter.diag().tolist())

        for idx, hg in enumerate(hg_te_list):
            hg["paths"] = []
            adj_iter = torch.eye(self.te_adj_list[idx].shape[0])
            for _ in range(self.n_length):
                adj_iter = adj_iter.mm(self.te_adj_list[idx])
                hg["paths"].append(adj_iter.diag().tolist())
        
        hg_all_list = hg_list + hg_te_list

        for idx in range(self.n_length):
            p_set = set()
            for hg in hg_all_list:
                for v in hg["paths"][idx]:
                    p_set.add(v)
            self.paths_set_list.append(p_set)


        for hg in hg_list:
            p_ft = []
            for idx in range(self.n_length):
                p_cnt = {k: 0 for k in sorted(self.paths_set_list[idx])}
                for v in hg["paths"][idx]:
                    p_cnt[v] += 1
                p_ft.extend([v for k, v in p_cnt.items()])
            self.path_ft.append(p_ft)

        self.path_ft = torch.tensor(self.path_ft)
        

        self._cnt = [defaultdict(int) for _ in range(len(hg_list))]
        self.remap_v(hg_list, self._cnt)
        self.remap_e(hg_list, self._cnt)
        for _ in range(self.n_iter):
            for hg in hg_list:
                tmp = []
                for e_idx in range(hg["dhg"].num_e):
                    cur_lbl = hg["e_lbl"][e_idx]
                    nbr_lbl = sorted(hg["v_lbl"][v_idx] for v_idx in hg["dhg"].nbr_v(e_idx))
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["e_lbl"] = tmp
            self.remap_e(hg_list, self._cnt)
            for hg in hg_list:
                tmp = []
                for v_idx in range(hg["dhg"].num_v):
                    cur_lbl = hg["v_lbl"][v_idx]
                    nbr_lbl = sorted(hg["e_lbl"][e_idx] for e_idx in hg["dhg"].nbr_e(v_idx))
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["v_lbl"] = tmp
            self.remap_v(hg_list, self._cnt)
        # 将计数转换为矩阵
        self.train_cnt = self.cnt2mat(self._cnt)
        self.train_id_ft = torch.cat([self.train_cnt.to_dense(), self.path_ft], dim=1)
        # 两两做内积
        self.train_ft = self.train_id_ft.mm(self.train_id_ft.t()).to_dense()
        if self.normalize:
            self.train_ft_diag = torch.diag(self.train_ft)
            self.train_ft = self.train_ft / torch.outer(self.train_ft_diag, self.train_ft_diag).sqrt()
            self.train_ft[torch.isnan(self.train_ft)] = 0
        return self.train_ft

    def transform(self, hg_list):
        adj_list = [hg["dhg"].H.mm(hg["dhg"].H.t()).to_dense() for hg in hg_list]
        path_ft = []

        for idx, hg in enumerate(hg_list):
            hg["paths"] = []
            adj_iter = torch.eye(adj_list[idx].shape[0])
            for _ in range(self.n_length):
                adj_iter = adj_iter.mm(adj_list[idx])
                hg["paths"].append(adj_iter.diag().tolist())

        for hg in hg_list:
            p_ft = []
            for idx in range(self.n_length):
                p_cnt = {k: 0 for k in sorted(self.paths_set_list[idx])}
                for v in hg["paths"][idx]:
                    p_cnt[v] += 1
                p_ft.extend([v for k, v in p_cnt.items()])
            path_ft.append(p_ft)

        path_ft = torch.tensor(path_ft)

        cnt = [defaultdict(int) for _ in range(len(hg_list))]
        self.remap_v(hg_list, cnt, drop=True)
        self.remap_e(hg_list, cnt, drop=True)
        for _ in range(self.n_iter):
            for hg in hg_list:
                tmp = []
                for e_idx in range(hg["dhg"].num_e):
                    cur_lbl = hg["e_lbl"][e_idx]
                    nbr_lbl = sorted(hg["v_lbl"][v_idx] for v_idx in hg["dhg"].nbr_v(e_idx))
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["e_lbl"] = tmp
            self.remap_e(hg_list, cnt, drop=True)
            for hg in hg_list:
                tmp = []
                for v_idx in range(hg["dhg"].num_v):
                    cur_lbl = hg["v_lbl"][v_idx]
                    nbr_lbl = sorted(hg["e_lbl"][e_idx] for e_idx in hg["dhg"].nbr_e(v_idx))
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["v_lbl"] = tmp
            self.remap_v(hg_list, cnt, drop=True)
        test_cnt = self.cnt2mat(cnt)
        test_id_ft = torch.cat([test_cnt.to_dense(), path_ft], dim=1)
        test_ft = test_id_ft.mm(self.train_id_ft.t()).to_dense()
        if self.normalize:
            test_ft_diag = torch.sparse.sum(test_cnt * test_cnt, dim=1).to_dense()
            test_ft = test_ft / torch.outer(test_ft_diag, self.train_ft_diag).sqrt()
            test_ft[torch.isnan(test_ft)] = 0
        return test_ft
