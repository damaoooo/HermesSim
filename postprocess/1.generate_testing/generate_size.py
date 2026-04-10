import argparse
import os
import json
import sys
import traceback
import pandas as pd
from os.path import join
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

PCODE_RAW_TRAIN_DS = join("dbs/Dataset-1/features/training/pcode_raw_Dataset-1_training")
PCODE_RAW_TEST_DS = join("dbs/Dataset-1/features/testing/pcode_raw_Dataset-1_testing")


def _safe_num_workers(num_workers):
    if num_workers is not None:
        return max(1, int(num_workers))
    return max(1, cpu_count() - 1)


def _process_one_file(fp):
    with open(fp, "r") as f:
        d = json.load(f)

    idb_path = list(d.keys())[0]
    d = d[idb_path]
    n = len(d)
    return {
        "ok": True,
        "idb_path": [idb_path] * n,
        "fva": list(d.keys()),
        "sizes": [len(f_data["SOG"]["nodes"]) for f_data in d.values()],
    }


def _process_one_file_safe(fp):
    try:
        return _process_one_file(fp)
    except Exception as e:
        return {
            "ok": False,
            "file": fp,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


def _write_failure_log(log_path, failures):
    if not failures:
        return
    with open(log_path, "w") as f:
        for item in failures:
            f.write(f"{item['file']}\n")
            f.write(f"{item['error']}\n")
            f.write(item["traceback"])
            if not item["traceback"].endswith("\n"):
                f.write("\n")
            f.write("\n")


def get_graph_sizes_data(dataset_path, num_workers=None, log_path=None):
    file_paths = [
        join(dataset_path, fn)
        for fn in os.listdir(dataset_path)
        if os.path.isfile(join(dataset_path, fn))
    ]
    graph_sizes = {
        "idb_path": [],
        "fva": [],
        "sizes": [],
    }
    failures = []

    num_workers = _safe_num_workers(num_workers)
    chunksize = max(1, len(file_paths) // (num_workers * 4)) if file_paths else 1
    with Pool(num_workers, maxtasksperchild=32) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_one_file_safe, file_paths, chunksize=chunksize),
            total=len(file_paths),
        ):
            if not result["ok"]:
                failures.append(result)
                continue
            graph_sizes["idb_path"].extend(result["idb_path"])
            graph_sizes["fva"].extend(result["fva"])
            graph_sizes["sizes"].extend(result["sizes"])

    if failures and log_path is not None:
        _write_failure_log(log_path, failures)

    if failures and not graph_sizes["idb_path"]:
        raise RuntimeError(f"All files failed while parsing {dataset_path}. See {log_path}")

    return graph_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_info_csv")
    parser.add_argument("--dataset-path", default=PCODE_RAW_TEST_DS)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()

    ds_info_csv = args.ds_info_csv
    log_path = ds_info_csv[:-4] + "_size_failures.log"
    d = pd.DataFrame(
        get_graph_sizes_data(args.dataset_path, num_workers=args.jobs, log_path=log_path),
        copy=False,
    )
    df = pd.read_csv(ds_info_csv).merge(
        d, how="left", on=["idb_path", "fva"], copy=False,
    )
    df.to_csv(ds_info_csv[:-4] + "_with_size.csv")
