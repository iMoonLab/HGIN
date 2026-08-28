#!/usr/bin/env bash
# Full reproduction: all 9 built-in datasets x {HG IA Subtree kernel, HGIN, IA-HGIN}.
# This takes several hours. For a quick check, use run_demo.sh instead.
# Protein datasets (Table VII in the paper) are NOT included: download the .pkl
# files first (see data/hypergraph/PROTEIN/readme.md) and then use protein_main.py.
set -u
PY=python

DATASETS="RHG_10 RHG_3 RHG_table RHG_pyramid IMDB_dir_form IMDB_dir_genre IMDB_wri_form IMDB_wri_genre steam_player"

echo "########## HG IA Subtree kernel (Tables III & IV) ##########"
for ds in $DATASETS; do
  echo "===== hypergraph_subtree_id / $ds ====="
  $PY ml_main.py data.name=$ds model.name=hypergraph_subtree_id 2>&1 | grep -E "mean test results"
done

echo "########## HGIN / IA-HGIN (Tables V & VI) ##########"
for model in hgin ia_hgin; do
  for ds in $DATASETS; do
    echo "===== $model / $ds ====="
    $PY dl_main.py data.name=$ds model.name=$model 2>&1 | grep -E "mean test results"
  done
done

echo "ALL DONE"
