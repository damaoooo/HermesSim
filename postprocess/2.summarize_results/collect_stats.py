#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#                                                                            #
#  Code for the USENIX Security '24 paper:                                   #
#  Code is not Natural Language: Unlock the Power of Semantics-Oriented      #
#             Graph Representation for Binary Code Similarity Detection      #
#                                                                            #
#  MIT License                                                               #
#                                                                            #
#  Copyright (c) 2023 SJTU NSSL Lab                                     #
#                                                                            #
##############################################################################

import os
import sys

import pandas as pd

from os.path import join, basename, dirname

from recall_mrr import compute_and_save_mrr_and_recall, form_result_csv_name
from compute_sim_df_for_jTrans import compute_sim_for_jTrans


def collect_for_one_task(pos_pair_csv, in_dir):
    rank_method = 'max'
    result_dfs = []
    result_csv = form_result_csv_name(basename(pos_pair_csv), rank_method)
    sim_df_csv = basename(pos_pair_csv)[:-4] + "_sim.csv"
    ds1_pkl_fn, x64_pkl_fn = "testing_Dataset-1.pkl", "x64_testing.pkl"
    for root, _, filenames in os.walk(in_dir):
        result_csv_fp = join(root, result_csv)
        if os.path.exists(result_csv_fp):
            result_dfs.append(pd.read_csv(result_csv_fp))
            continue
        # Three modes (pair sim csv, full dataset pkl, x64-sub dataset pkl)
        if sim_df_csv in filenames:
            compute_and_save_mrr_and_recall(
                root, pos_pair_csv, None, rank_method)
        elif ds1_pkl_fn in filenames:
            pkl_fp = join(root, ds1_pkl_fn)
            compute_and_save_mrr_and_recall(
                root, pos_pair_csv, pkl_fp, rank_method)
        elif x64_pkl_fn in filenames and '-arch_x-bit_64' in pos_pair_csv:
            compute_sim_for_jTrans(join(root, x64_pkl_fn), pos_pair_csv, root)
            neg_pair_csv = pos_pair_csv.replace("pos-", "neg-")
            compute_sim_for_jTrans(join(root, x64_pkl_fn), neg_pair_csv, root)
            compute_and_save_mrr_and_recall(
                root, pos_pair_csv, None, rank_method)
        if os.path.exists(result_csv_fp):
            result_dfs.append(pd.read_csv(result_csv_fp))
    if not result_dfs:
        print(f"[W] No result files found for {pos_pair_csv} under {in_dir}. Skip. ")
        return False
    mrr_summary = pd.concat(result_dfs)
    mrr_summary.to_csv(join(in_dir, "summary_" + result_csv))
    return True


def collect_for_all_tasks(pair_dir, in_dir):
    if not os.path.isdir(in_dir):
        print(f"[W] Result directory {in_dir} does not exist. Nothing to collect. ")
        return
    num_tasks, num_summaries = 0, 0
    for root, _, filenames in os.walk(pair_dir):
        for fn in filenames:
            if fn.startswith("pos-") and fn.endswith(".csv"):
                num_tasks += 1
                num_summaries += int(collect_for_one_task(join(root, fn), in_dir))
    if num_tasks == 0:
        print(f"[W] No pos-*.csv found under {pair_dir}. ")
    elif num_summaries == 0:
        print(f"[W] No task summaries were generated from {num_tasks} task(s). ")


if __name__ == '__main__':
    pos_pair_csv_or_dir = sys.argv[1]
    target_dir = sys.argv[2]
    if os.path.isdir(pos_pair_csv_or_dir):
        print(f"[D]: Collect results of all tasks in {pos_pair_csv_or_dir}. ")
        collect_for_all_tasks(pos_pair_csv_or_dir, target_dir)
    elif os.path.exists(pos_pair_csv_or_dir):
        print(f"[D]: Collect for the task {pos_pair_csv_or_dir}. ")
        collect_for_one_task(pos_pair_csv_or_dir, target_dir)
    else:
        print(f"Error: {pos_pair_csv_or_dir} does not exists. ")
