import dhg
import hydra
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

from models import (
    HypergraphRootedKernel,
    GraphSubtreeKernel,
    GraphletSampling,
    HypergraphDirectedLineKernel,
    HypergraphSubtreeKernel,
    HypergraphWLEdgeKernel,
    HypergraphSubtreeIDKernel
)
from utils import load_data, separate_data
from utils import train_infer_MLP, train_infer_SVM

print = logging.info
multi_label, criterion = None, None


@hydra.main(config_path=".", config_name="ml_config", version_base=None)
def main(cfg: DictConfig):
    if cfg.model.name in [
        "hypergraph_rooted",
        "hypergraph_directed_line",
        "hypergraph_subtree",
        "hypergraph_subtree_v",
        "hypergraph_subtree_e",
        "hypergraph_wl_e",
        "hypergraph_subtree_id",
    ]:
        model_type = "hypergraph"
    else:
        model_type = "graph"
    print(OmegaConf.to_yaml(cfg))
    global multi_label, criterion
    dhg.random.set_seed(cfg.seed)
    x_list, y_list, meta = load_data(cfg.data.name, cfg.data.root, cfg.data.degree_as_tag, model_type)
    multi_label = meta["multi_label"]
    n_classes = meta["n_classes"]

    n_fold_idx = separate_data(x_list, y_list, cfg.data.n_fold, cfg.seed)

    if cfg.model.name == "graph_subtree":
        model = GraphSubtreeKernel(normalize=cfg.model.normalize)
    elif cfg.model.name == "graphlet_sampling":
        model = GraphletSampling(normalize=cfg.model.normalize, sampling={})
    elif cfg.model.name == "hypergraph_rooted":
        model = HypergraphRootedKernel(normalize=cfg.model.normalize)
    elif cfg.model.name == "hypergraph_directed_line":
        model = HypergraphDirectedLineKernel(normalize=cfg.model.normalize)
    elif cfg.model.name == "hypergraph_subtree":
        model = HypergraphSubtreeKernel(normalize=cfg.model.normalize)
    elif cfg.model.name == "hypergraph_subtree_v":
        model = HypergraphSubtreeKernel(way="v", normalize=cfg.model.normalize)
    elif cfg.model.name == "hypergraph_subtree_e":
        model = HypergraphSubtreeKernel(way="e", normalize=cfg.model.normalize)
    elif cfg.model.name == "hypergraph_wl_e":
        model = HypergraphWLEdgeKernel(normalize=cfg.model.normalize)
    elif cfg.model.name == "hypergraph_subtree_id":
        model = HypergraphSubtreeIDKernel(normalize=cfg.model.normalize)
    else:
        raise NotImplementedError

    test_res, test_all_res = [], defaultdict(list)
    for fold_idx, (train_idx, test_idx) in enumerate(n_fold_idx):

        _x_list, _y_list = deepcopy(x_list), deepcopy(y_list)
        train_x_list, train_y_list, test_x_list, test_y_list = [], [], [], []
        for idx in train_idx:
            train_x_list.append(_x_list[idx])
            train_y_list.append(_y_list[idx])
        for idx in test_idx:
            test_x_list.append(_x_list[idx])
            test_y_list.append(_y_list[idx])

        train_y, test_y = np.array(train_y_list), np.array(test_y_list)
        if cfg.model.name == "hypergraph_subtree_id":
            K_train = model.fit_transform(train_x_list, test_x_list).cpu().numpy()
        else:
            K_train = model.fit_transform(train_x_list).cpu().numpy()
        K_test = model.transform(test_x_list).cpu().numpy()

        # --------------------------------------------------------------
        # MLP
        # test_val, best_res = train_infer_MLP(K_train, train_y, K_test, test_y, n_classes, multi_label, cfg.device)
        # SVM
        test_val, best_res = train_infer_SVM(K_train, train_y, K_test, test_y, multi_label)
        # ---------------------------------------------------------------
        print(f"[{fold_idx+1}/{len(n_fold_idx)}] test results: {test_val:.4f}")
        test_res.append(test_val)
        for k, v in best_res.items():
            test_all_res[k].append(v)
    res = {k: sum(v) / len(v) for k, v in test_all_res.items()}
    print(f"mean test results: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])}")
    return test_res


if __name__ == "__main__":
    main()
