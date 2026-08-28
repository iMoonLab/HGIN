#!/usr/bin/env bash
# Quick demo: reproduce selected results from the TPAMI paper
# "How Powerful are Hypergraph Neural Networks?"
# Total runtime: ~35 min on a single GPU (RTX 2080 Ti or similar).
# Expected results (single seed 2024, 5-fold CV) are listed next to each run;
# the paper reports the mean over 5 seeds, so small deviations are normal.
set -u
PY=python

echo "########## [1/6] HG IA Subtree kernel / IMDB-Wri-Form (~15 s) ##########"
echo "# expected acc: 51.33"
$PY ml_main.py data.name=IMDB_wri_form model.name=hypergraph_subtree_id 2>&1 | grep -E "mean test results"

echo "########## [2/6] HG IA Subtree kernel / RHG-10 (~3 min) ##########"
echo "# expected acc: 97.85"
$PY ml_main.py data.name=RHG_10 model.name=hypergraph_subtree_id 2>&1 | grep -E "mean test results"

echo "########## [3/6] HG IA Subtree kernel / IMDB-Dir-Form (~4 min) ##########"
echo "# expected acc: 67.42"
$PY ml_main.py data.name=IMDB_dir_form model.name=hypergraph_subtree_id 2>&1 | grep -E "mean test results"

echo "########## [4/6] HGIN / IMDB-Wri-Form (~2 min) ##########"
echo "# expected acc: 51.98"
$PY dl_main.py data.name=IMDB_wri_form model.name=hgin 2>&1 | grep -E "mean test results"

echo "########## [5/6] HGIN / RHG-10 (~13 min) ##########"
echo "# expected acc: 97.78"
$PY dl_main.py data.name=RHG_10 model.name=hgin 2>&1 | grep -E "mean test results"

echo "########## [6/6] IA-HGIN / RHG-3 (~11 min) ##########"
echo "# expected acc: 99.92"
$PY dl_main.py data.name=RHG_3 model.name=ia_hgin 2>&1 | grep -E "mean test results"

echo "DEMO DONE"
