import dhg
import hydra
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

from utils import GraphLoader, HypergraphLoader
from utils import load_data, separate_data, performance
from models import HGIN

print = logging.info
multi_label, criterion = None, None
g2hg_func = dhg.Hypergraph.from_graph
hg2g_func = dhg.Graph.from_hypergraph_clique

def train(train_data, model, optimizer, epoch, device):
    model.train()

    for batch_idx, (data, target) in enumerate(train_data):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_data)}] -> loss: {loss.item():.6f}")


def test(test_data, model, device):
    model.eval()
    outputs, targets = [], []
    for data, target in test_data:
        output = model(data)
        outputs.append(output.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets)
    val, res = performance(outputs, targets, multi_label)
    print(f"----> test res: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])} \n")
    return val, res


@hydra.main(config_path=".", config_name="dl_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    global multi_label, criterion
    dhg.random.set_seed(cfg.seed)
    device = cfg.device
    
    model_type = "hypergraph"
    model_to_use = HGIN
    if cfg.model.name == "ia_hgin":
        x_list, y_list, meta = load_data(cfg.data.name, cfg.data.root, cfg.data.degree_as_tag, model_type, cfg.model.name)
    else:
        x_list, y_list, meta = load_data(cfg.data.name, cfg.data.root, cfg.data.degree_as_tag, model_type)
    multi_label = meta["multi_label"]
    ft_dim = meta["ft_dim"]
    n_classes = meta["n_classes"]

    n_fold_idx = separate_data(x_list, y_list, cfg.data.n_fold, cfg.seed)

    if multi_label:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    test_res, test_all_res = [], defaultdict(list)
    for fold_idx, (train_idx, test_idx) in enumerate(n_fold_idx):
        best_test_val = 0
        
        if cfg.model.name == "ia_hgin":
            model = model_to_use(ft_dim, cfg.model.hidden_dim, n_classes, multi_label, IA_dim=11).to(cfg.device)
        else:
            model = model_to_use(ft_dim, cfg.model.hidden_dim, n_classes, multi_label).to(cfg.device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        train_x_list, train_y_list, test_x_list, test_y_list = [], [], [], []
        for idx in train_idx:
            train_x_list.append(x_list[idx])
            train_y_list.append(y_list[idx])
        for idx in test_idx:
            test_x_list.append(x_list[idx])
            test_y_list.append(y_list[idx])

        if model_type == "graph":
            train_data = GraphLoader(
                train_x_list, train_y_list, cfg.data.batch_size, cfg.data.num_workers, True, cfg.device, hg2g_func
            )
            test_data = GraphLoader(
                test_x_list, test_y_list, cfg.data.batch_size, cfg.data.num_workers, False, cfg.device, hg2g_func
            )
        else:
            train_data = HypergraphLoader(
                train_x_list, train_y_list, cfg.data.batch_size, cfg.data.num_workers, True, cfg.device, g2hg_func
            )
            test_data = HypergraphLoader(
                test_x_list, test_y_list, cfg.data.batch_size, cfg.data.num_workers, False, cfg.device, g2hg_func
            )

        for epoch in range(1, cfg.optim.max_epoch + 1):
            print(f"[{fold_idx+1}/{len(n_fold_idx)}]")
            train(train_data, model, optimizer, epoch, device)
            test_val, res = test(test_data, model, device)
            if test_val > best_test_val:
                best_test_val = test_val
                best_res = res
            scheduler.step()
        print(f"[{fold_idx+1}/{len(n_fold_idx)}] test results: {best_test_val:.4f}\n\n")
        test_res.append(best_test_val)
        for k, v in best_res.items():
            test_all_res[k].append(v)
    res = {k: sum(v) / len(v) for k, v in test_all_res.items()}
    print(f"mean test results: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])}")
    print(f"model:{cfg.model.name} ' | ' seed:{cfg.seed} ' | ' dataset:{cfg.data.name}")
    return test_res


if __name__ == "__main__":
    main()
