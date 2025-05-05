import os
import json
import sys
import pandas as pd
from os.path import join
from tqdm import tqdm

PCODE_RAW_TRAIN_DS = join("dbs/Dataset-1/features/training/pcode_raw_Dataset-1_training")
PCODE_RAW_TEST_DS = join("dbs/Dataset-1/features/testing/pcode_raw_Dataset-1_testing")

def get_graph_sizes_data(dataset_path):
    graph_sizes = {
        "idb_path": [],
        "fva": [],
        "sizes": [], 
    }
    for fn in tqdm(os.listdir(dataset_path)):
        fp = join(dataset_path, fn)
        with open(fp, "r") as f:
            try:
                d = json.load(f)
            except Exception as e:
                print(fp)
                print(e)
        idb_path = list(d.keys())[0]
        d = d[idb_path]
        n = len(d)
        graph_sizes["idb_path"].extend([idb_path] * n)
        for fva, f_data in d.items():
            graph_sizes["fva"].append(fva)
            graph_sizes["sizes"].append(len(f_data['SOG']['nodes']))
    return graph_sizes

if __name__ == "__main__":
    ds_info_csv = sys.argv[1]
    d = pd.DataFrame(get_graph_sizes_data(PCODE_RAW_TEST_DS), copy=False)
    df = pd.read_csv(ds_info_csv).merge(
        d, how="left", on=["idb_path", "fva"], copy=False, 
    )
    df.to_csv(ds_info_csv[:-4]+"_with_size.csv")