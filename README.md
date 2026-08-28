<div align="center" id="top">

<img src="docs/logo.svg" alt="HGIN" width="30%"/>

### *How Powerful are Hypergraph Neural Networks?*

**Yifan Feng · Rizhuo Huang · Yifan Zhang · Shaoyi Du · Shihui Ying · Zongze Wu · Yue Gao\***

<a href="https://www.computer.org/csdl/journal/tp/5555/01/11657965/2j3auScOCuQ"><img alt="IEEE TPAMI 2026" src="https://img.shields.io/badge/IEEE-TPAMI%202026-4F46E5?style=flat-square"></a>
<a href="https://doi.org/10.1109/TPAMI.2026.3724944"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1109%2FTPAMI.2026.3724944-2DD4BF?style=flat-square"></a>
<a href="LICENSE.txt"><img alt="License" src="https://img.shields.io/badge/License-MIT-8B83F6?style=flat-square"></a>

</div>

<br>

WL-style kernels and GNNs capture neighborhood connectivity, but miss the higher-order structures that make hypergraphs powerful. This work closes the gap with one kernel and two networks:

- **HG IA Subtree Kernel** — hypergraph WL subtree + **closed-path counts** of varying lengths; distinguishes uniform-regular hypergraphs that WL kernels cannot.
- **HGIN** — vertex → hyperedge → vertex message passing with MLP aggregation, matching the HG WL subtree kernel's expressive power.
- **IA-HGIN** — HGIN + a dedicated channel for closed-path distribution features. Strictly stronger.

## Installation

```bash
git clone https://github.com/iMoonLab/HGIN && cd HGIN
uv venv --python 3.10 && source .venv/bin/activate   # Windows: .venv/Scripts/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # CPU: uv pip install torch
uv pip install -r requirements.txt
```

## Reproduce in one command

```bash
bash run_demo.sh      # 6 representative experiments, ~35 min on one GPU
```

| Experiment | Command | Acc |
|---|---|---|
| HG IA Subtree / IMDB-Wri-Form (~15 s) | `python ml_main.py data.name=IMDB_wri_form model.name=hypergraph_subtree_id` | 51.33 |
| HG IA Subtree / RHG-10 (~3 min) | `python ml_main.py data.name=RHG_10 model.name=hypergraph_subtree_id` | 97.85 |
| HG IA Subtree / IMDB-Dir-Form (~4 min) | `python ml_main.py data.name=IMDB_dir_form model.name=hypergraph_subtree_id` | 67.42 |
| HGIN / IMDB-Wri-Form (~2 min) | `python dl_main.py data.name=IMDB_wri_form model.name=hgin` | 51.98 |
| HGIN / RHG-10 (~13 min) | `python dl_main.py data.name=RHG_10 model.name=hgin` | 97.78 |
| IA-HGIN / RHG-3 (~11 min) | `python dl_main.py data.name=RHG_3 model.name=ia_hgin` | 99.92 |

<sub>Results are averaged over 5 seeds; the demo runs a single seed with 5-fold CV, so small deviations are expected.</sub>

<details>
<summary><b>Full reproduction</b> — all 9 datasets × {kernel, HGIN, IA-HGIN} (Tables III–VI, several hours)</summary>

```bash
bash run_full_sweep.sh
```

Single experiments are configured via [Hydra](https://hydra.cc/):

```bash
python ml_main.py data.name=<dataset> model.name=<kernel>   # kernel-based methods
python dl_main.py data.name=<dataset> model.name=hgin       # or ia_hgin
```

`<dataset>`: `RHG_10`, `RHG_3`, `RHG_table`, `RHG_pyramid`, `IMDB_dir_form`, `IMDB_dir_genre`, `IMDB_wri_form`, `IMDB_wri_genre`, `steam_player` · `<kernel>`: `hypergraph_subtree_id`, `hypergraph_subtree(_v|_e)`, `hypergraph_wl_e`, `hypergraph_rooted`, `hypergraph_directed_line`, `graph_subtree`, `graphlet_sampling`

</details>

<details>
<summary><b>Protein datasets</b> (Table VII) — manual download required</summary>

Download `EnzymeClass.pkl`, `ProteinFamily.pkl`, `StructuralClass_TP.pkl`, `StructuralClass_CL.pkl` into `data/hypergraph/PROTEIN/`:

- Baidu Netdisk: https://pan.baidu.com/s/162T8QhftOgVubQUONe0ufA?pwd=d89r (code: `d89r`)

```bash
python protein_main.py data.name=EnzymeClass model.name=ia_hgin
```

</details>

<details>
<summary><b>Citation</b></summary>

```bibtex
@article{feng2026how,
  title   = {How Powerful are Hypergraph Neural Networks?},
  author  = {Feng, Yifan and Huang, Rizhuo and Zhang, Yifan and Du, Shaoyi and Ying, Shihui and Wu, Zongze and Gao, Yue},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026},
  doi     = {10.1109/TPAMI.2026.3724944}
}
```

</details>

<br>

<div align="center">
<sub>MIT License · Made with ❤️ by <a href="https://fengyifan.site">Yifan Feng</a> and Rizhuo Huang</sub>
</div>
