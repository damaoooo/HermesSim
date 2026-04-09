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

import click
import json
import numpy as np
import os
import pickle
import traceback
import shutil
import gc

from typing import Dict
import multiprocessing

from collections import Counter
from collections import defaultdict
from tqdm import tqdm

GRAPH_TYPES = ['ISCG','TSCG','SOG']


def make_counter_bucket():
    counters = {'opc': Counter(), 'val': Counter()}
    for arch in ['mips', 'arm', 'x']:
        counters[f'{arch}_reg'] = Counter()
    return counters


def get_worker_count(num_tasks, reserve_one_cpu=True):
    if num_tasks <= 0:
        return 1
    cpu_total = multiprocessing.cpu_count()
    workers = cpu_total - 1 if reserve_one_cpu and cpu_total > 1 else cpu_total
    workers = max(1, workers)
    return min(workers, num_tasks)


def get_chunksize(num_tasks, num_workers):
    if num_tasks <= 0:
        return 1
    return max(1, num_tasks // (num_workers * 4))


def merge_counter_buckets(dst, src):
    for gtype, counters in src.items():
        for ty, counter in counters.items():
            dst[gtype][ty].update(counter)


def get_failure_log_path(output_dir, dataset, stage):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    safe_dataset = dataset if dataset else "unknown_dataset"
    return os.path.join(log_dir, f"{safe_dataset}_{stage}_failures.log")


def write_failure_log(output_dir, dataset, stage, failures):
    log_path = get_failure_log_path(output_dir, dataset, stage)
    if not failures:
        if os.path.exists(log_path):
            os.remove(log_path)
        return None

    with open(log_path, "w") as f_out:
        f_out.write(f"stage: {stage}\n")
        f_out.write(f"dataset: {dataset}\n")
        f_out.write(f"failures: {len(failures)}\n\n")
        for idx, failure in enumerate(failures, start=1):
            f_out.write(f"[{idx}] file: {failure['json_path']}\n")
            f_out.write(f"error: {failure['error']}\n")
            f_out.write("traceback:\n")
            f_out.write(failure["traceback"].rstrip())
            f_out.write("\n")
            f_out.write("=" * 80)
            f_out.write("\n")
    return log_path


def dump_pickle_file(obj, output_path):
    with open(output_path, "wb") as f_out:
        pickle.dump(obj, f_out)


def load_pickle_file(input_path):
    with open(input_path, "rb") as f_in:
        return pickle.load(f_in)


def reset_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def remove_paths(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def get_graph_output_name(freq_mode, dump_kind):
    suffix = "json" if dump_kind == "str" else "pkl"
    return f"graph_func_dict_opc_{freq_mode}.{suffix}"


def get_graph_output_path(output_dir, gtype, dataset, freq_mode, dump_kind):
    sub_dir = get_sub_dir(output_dir, gtype, dataset)
    return os.path.join(sub_dir, get_graph_output_name(freq_mode, dump_kind))


def get_shard_dir(output_dir, gtype, dataset, freq_mode, dump_kind):
    sub_dir = get_sub_dir(output_dir, gtype, dataset)
    output_name = get_graph_output_name(freq_mode, dump_kind)
    return os.path.join(sub_dir, f".{output_name}.shards")


def prepare_shard_dirs(output_dir, dataset, freq_mode, dump_str, dump_pkl):
    shard_dirs = {"str": {}, "pkl": {}}
    if dump_str:
        for gtype in GRAPH_TYPES:
            shard_dir = get_shard_dir(output_dir, gtype, dataset, freq_mode, "str")
            reset_dir(shard_dir)
            shard_dirs["str"][gtype] = shard_dir
    if dump_pkl:
        for gtype in GRAPH_TYPES:
            shard_dir = get_shard_dir(output_dir, gtype, dataset, freq_mode, "pkl")
            reset_dir(shard_dir)
            shard_dirs["pkl"][gtype] = shard_dir
    return shard_dirs


def cleanup_shard_dirs(shard_dirs):
    for shard_kind in shard_dirs.values():
        for shard_dir in shard_kind.values():
            if os.path.exists(shard_dir):
                shutil.rmtree(shard_dir)

def parse_nxopr(pcode_asm):
    s = pcode_asm.find('(')
    e = pcode_asm.find(')')
    return tuple(pcode_asm[s+1:e].split(', ')), pcode_asm[e+1:].strip()


def parse_pcode(pcode_asm):
    '''
    Examples: 
    (register, 0x20, 4) COPY (const, 0x0, 4)
    (unique, 0x8380, 4) INT_ADD (register, 0x4c, 4) , (const, 0xfffffff0, 4)
     ---  STORE (STORE, 0x1a1, 0) , (unique, 0x8280, 4) , (register, 0x20, 4)
     ---  BRANCH (ram, 0x22128, 4)
    '''
    NOP_OPERAND = ' --- '
    dst_opr = None
    if pcode_asm.startswith(NOP_OPERAND):
        pcode_asm = pcode_asm[len(NOP_OPERAND)+1:]
    else:
        dst_opr, pcode_asm = parse_nxopr(pcode_asm)
    opc_e = pcode_asm.find(' ')
    if opc_e != -1:
        opc, pcode_asm = pcode_asm[:opc_e], pcode_asm[opc_e:].strip()
    else:
        opc, pcode_asm = pcode_asm, ""
    oprs = [] if dst_opr is None else [dst_opr,]
    while len(pcode_asm) != 0:
        src_opr, pcode_asm = parse_nxopr(pcode_asm)
        oprs.append(src_opr)
    return (opc, oprs)


def normalize_pcode_opr(opr, arch):
    if opr[0] in ['register']:
        return f'{arch}_reg', arch + '_' + '_'.join(opr)
    elif opr[0] in ['STORE', 'const']:
        return 'val', '_'.join(opr[:-1]) # omit dummy size field
    elif opr[0] in ['unique', 'NewUnique', 'ram', 'stack', 'VARIABLE']:
        return 'val', opr[0]
    else:
        raise Exception(f"Unkown operand type {opr[0]}. FULL: {opr}. ")


def normalize_pcode(pcode, arch):
    normalized_pcode = [('opc', pcode[0])]
    for opr in pcode[1]:
        normalized_pcode.append(normalize_pcode_opr(opr, arch))
    return normalized_pcode


def normalize_sng_opc(opc, arch):
    # Raw Formats:
    # ConstLong: L(%x, %d)
    # ConstDouble: D(%f, %d)
    # MemorySpace: SPACE(%x)
    # Store: "%s(%x, %d)", {'REG','MEM','STA','OTH'}, id, size
    # Project/Subpiece: PROJ(%d)
    # 1. Retain small integers.
    # 2. Register regs with arch id.
    if opc.startswith('L('):
        v = int(opc[2:opc.find(',')], 16)
        return 'val', f'L_{v}'
    elif opc.startswith('D('):
        v = float(opc[2:opc.find(',')])
        return 'val', f'D_{v}'
    elif opc.startswith('REG('):
        v = int(opc[4:opc.find(',')], 16)
        s = int(opc[opc.find(',')+1:opc.find(')')], 16)
        return f'{arch}_reg', f'{arch}_REG_{v}_{s}'
    elif opc[:3] in ['MEM', 'STA', 'OTH']:
        return 'val', opc[:3]
    elif opc.startswith('PROJ('):
        return 'opc', 'PROJ'
    else:
        return 'opc', opc


def process_nverb(gtype, nverb: list, arch):
    assert isinstance(nverb, list)
    if len(nverb) == 0:
        return []
    if gtype == 'SOG':
        ty, opc = normalize_sng_opc(nverb[0], arch)
        return [(ty, opc),]
    elif gtype == 'ISCG':
        parsed = parse_pcode(nverb[0])
        return normalize_pcode(parsed, arch)
    elif gtype == 'ACFG':
        results = []
        for inst in nverb:
            parsed = parse_pcode(inst)
            oprs = normalize_pcode(parsed, arch)
            results.extend(oprs)
        return results
    elif gtype == 'TSCG':
        if '(' == nverb[0][0]:
            ty, opc = normalize_pcode_opr(parse_nxopr(nverb[0])[0], arch)
        else:
            ty, opc = 'opc', nverb[0]
        return [(ty, opc),]
    else:
        assert "Unkown Graph Type"


def token_mapping_map(args):
    input_folder, f_json, not_cached_graph_types, any_cached = args
    json_path = os.path.join(input_folder, f_json)
    try:
        num_func = 0
        opc_counters = {
            gtype: make_counter_bucket() for gtype in not_cached_graph_types
        }
        opc_occurs = {
            gtype: make_counter_bucket() for gtype in not_cached_graph_types
        }

        with open(json_path) as f_in:
            jj = json.load(f_in)

        arch = f_json.split('-')[0][:-2]
        idb_path = list(jj.keys())[0]
        j_data = jj[idb_path]
        for key in ['arch']:
            if key in j_data:
                del j_data[key]

        # Iterate over each function
        for fva in j_data:
            for gtype in not_cached_graph_types:
                opc_sets = defaultdict(set)
                fva_data = j_data[fva][gtype]
                # Iterate over each basic-block
                for bb in fva_data['nverbs']:
                    nverb = fva_data['nverbs'][bb]
                    for ty, opc in process_nverb(gtype, nverb, arch):
                        opc_counters[gtype][ty].update([opc])
                        opc_sets[ty].add(opc)
                for ty, opc_set in opc_sets.items():
                    opc_occurs[gtype][ty].update(opc_set)
        if not any_cached:
            num_func += len(j_data)

        result = {
            "num_func": num_func,
            "opc_counters": opc_counters,
            "opc_occurs": opc_occurs,
        }
        return {"ok": True, "json_path": json_path, "result": result}
    except Exception as exc:
        return {
            "ok": False,
            "json_path": json_path,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def token_mapping_parallel(input_folder, output_dir, dataset, freq_mode=True):
    print("[i] Freq_mode: ", freq_mode)
    idmaps, opc_counters, opc_occurs = {}, {}, {}
    cached = {}
    num_func = 0

    # Try loading caches
    for gtype in GRAPH_TYPES:
        sub_dir = get_sub_dir(output_dir, gtype)
        counter_path = os.path.join(sub_dir, "opc_counter.json")
        occurs_path = os.path.join(sub_dir, "opc_occurs.json")
        if os.path.exists(counter_path) and os.path.exists(occurs_path):
            with open(counter_path, "r") as f:
                opc_counters[gtype] = json.load(f)
            with open(occurs_path, "r") as f:
                opc_occurs[gtype] = json.load(f)
                num_func = opc_occurs[gtype]["num_funcs"]
            cached[gtype] = True
        else:
            cached[gtype] = False

    opc_counters: Dict[str, Dict[str, Counter]]
    opc_occurs: Dict[str, Dict[str, Counter]]

    any_cached = sum(cached[gtype] for gtype in GRAPH_TYPES) != 0
    not_cached_graph_types = list(g for g in GRAPH_TYPES if not cached[g])

    if len(not_cached_graph_types) != 0:
        for gtype in not_cached_graph_types:
            opc_counters[gtype] = make_counter_bucket()
            opc_occurs[gtype] = make_counter_bucket()

        failures = []
        json_files = sorted(
            f_json for f_json in os.listdir(input_folder) if f_json.endswith(".json")
        )
        worker_count = get_worker_count(len(json_files))
        chunksize = get_chunksize(len(json_files), worker_count)
        print(f"[i] Token Mapping workers: {worker_count} (chunksize={chunksize})")

        tasks = [
            (input_folder, f_json, tuple(not_cached_graph_types), any_cached)
            for f_json in json_files
        ]
        success_count = 0
        with multiprocessing.Pool(processes=worker_count, maxtasksperchild=32) as pool:
            for response in tqdm(
                pool.imap_unordered(token_mapping_map, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="Token Mapping",
                dynamic_ncols=True,
            ):
                if not response["ok"]:
                    failures.append(response)
                    continue
                success_count += 1
                result = response["result"]
                num_func += result["num_func"]
                merge_counter_buckets(opc_counters, result["opc_counters"])
                merge_counter_buckets(opc_occurs, result["opc_occurs"])

        failure_log = write_failure_log(output_dir, dataset, "token_mapping", failures)
        if failure_log is not None:
            print(f"[W] Token mapping failed for {len(failures)} file(s). See {failure_log}")
        if tasks and success_count == 0:
            raise RuntimeError(
                f"Token mapping failed for all input files. See {failure_log}"
            )

        # Cache results
        for gtype in not_cached_graph_types:
            sub_dir = get_sub_dir(output_dir, gtype)
            output_path = os.path.join(sub_dir, "opc_counter.json")
            with open(output_path, "w") as f:
                json.dump(opc_counters[gtype], f)
            output_path = os.path.join(sub_dir, "opc_occurs.json")
            opc_occurs[gtype]["num_funcs"] = num_func
            with open(output_path, "w") as f:
                json.dump(opc_occurs[gtype], f)

                
    # Assigning each word an ID
    print(f"num funcs: {num_func}")
    for gtype in GRAPH_TYPES:
        idmaps[gtype] = {'padding': 0} if gtype in ['ISCG', 'ACFG'] else {}
    for gtype, opc_cnts in opc_counters.items():
        print(f"[D] Processing {gtype}. ")
        ths = dict([
            ('opc', 55 if gtype != 'ACFG' else 50), 
            ('val', 0.01),    
            *[(f'{arch}_reg', 0.01) for arch in ['mips', 'arm', 'x']],
        ]) # thresholds
        for ty, opc_cnt in opc_cnts.items():
            if ty.endswith("_occur"):
                continue
            if not isinstance(opc_cnt, dict):
                opc_cnt = [(k,v) for k,v in opc_cnt.most_common()]
            else:
                opc_cnt = sorted(list(opc_cnt.items()), key=lambda k:k[1], reverse=True)
            mapped_cnt = 0
            tot_cnt = sum([v for _, v in opc_cnt])
            idmaps[gtype][ty] = len(idmaps[gtype])
            start_id = len(idmaps[gtype])
            for i, (k, v) in enumerate(opc_cnt):
                idmaps[gtype][k] = i + start_id
                mapped_cnt += v
                if isinstance(ths[ty], float):
                    if not freq_mode and mapped_cnt / tot_cnt > ths[ty]:
                        break
                    elif freq_mode and v / num_func < ths[ty]:
                        break
                elif isinstance(ths[ty], int) and i + 1 >= ths[ty]:
                    break
            print("[D] Found: {} mnemonics.".format(len(opc_cnt)))
            print("[D] Num of mnemonics mapped: {}".format(len(idmaps[gtype])-start_id))
        print("[D] Tot Num of mnemonics mapped: {}".format(len(idmaps[gtype])))
    return idmaps


def token_mapping(input_folder, output_dir, freq_mode=True):
    print("[i] Freq_mode: ", freq_mode)
    idmaps, opc_counters, opc_occurs = {}, {}, {}
    cached = {}
    num_func = 0

    # Try loading caches
    for gtype in GRAPH_TYPES:
        sub_dir = get_sub_dir(output_dir, gtype)
        counter_path = os.path.join(sub_dir, "opc_counter.json")
        occurs_path = os.path.join(sub_dir, "opc_occurs.json")
        if os.path.exists(counter_path) and os.path.exists(occurs_path):
            with open(counter_path, "r") as f:
                opc_counters[gtype] = json.load(f)
            with open(occurs_path, "r") as f:
                opc_occurs[gtype] = json.load(f)
                num_func = opc_occurs[gtype]["num_funcs"]
            cached[gtype] = True
        else:
            opc_counters[gtype], opc_occurs[gtype] = (dict([
                ('opc', Counter()),
                ('val', Counter()),
                *[(f'{arch}_reg', Counter()) for arch in ['mips', 'arm', 'x']],
            ]) for _ in range(2))
            cached[gtype] = False

    opc_counters: Dict[str, Dict[str, Counter]]
    opc_occurs: Dict[str, Dict[str, Counter]]

    any_cached = sum(cached[gtype] for gtype in GRAPH_TYPES) != 0
    not_cached_graph_types = list(g for g in GRAPH_TYPES if not cached[g])

    if len(not_cached_graph_types) != 0:
        # Collect opc stats info
        for f_json in tqdm(os.listdir(input_folder), desc="Token Mapping"):
            if not f_json.endswith(".json"):
                continue

            json_path = os.path.join(input_folder, f_json)
            with open(json_path) as f_in:
                jj = json.load(f_in)

            arch = f_json.split('-')[0][:-2]
            idb_path = list(jj.keys())[0]
            j_data = jj[idb_path]
            for key in ['arch']:
                if key in j_data:
                    del j_data[key]

            # Iterate over each function
            for fva in j_data:
                for gtype in not_cached_graph_types:
                    opc_sets = defaultdict(set)
                    fva_data = j_data[fva][gtype]
                    # Iterate over each basic-block
                    for bb in fva_data['nverbs']:
                        nverb = fva_data['nverbs'][bb]
                        for ty, opc in process_nverb(gtype, nverb, arch):
                            opc_counters[gtype][ty].update([opc])
                            opc_sets[ty].add(opc)
                    for ty, opc_set in opc_sets.items():
                        opc_occurs[gtype][ty].update(opc_set)
            if not any_cached:
                num_func += len(j_data)

        # Cache results
        for gtype in not_cached_graph_types:
            sub_dir = get_sub_dir(output_dir, gtype)
            output_path = os.path.join(sub_dir, "opc_counter.json")
            with open(output_path, "w") as f:
                json.dump(opc_counters[gtype], f)
            output_path = os.path.join(sub_dir, "opc_occurs.json")
            opc_occurs[gtype]["num_funcs"] = num_func
            with open(output_path, "w") as f:
                json.dump(opc_occurs[gtype], f)

    # Assigning each word an ID
    print(f"num funcs: {num_func}")
    for gtype in GRAPH_TYPES:
        idmaps[gtype] = {'padding': 0} if gtype in ['ISCG', 'ACFG'] else {}
    for gtype, opc_cnts in opc_counters.items():
        print(f"[D] Processing {gtype}. ")
        ths = dict([
            ('opc', 55 if gtype != 'ACFG' else 50), 
            ('val', 0.01),    
            *[(f'{arch}_reg', 0.01) for arch in ['mips', 'arm', 'x']],
        ]) # thresholds
        for ty, opc_cnt in opc_cnts.items():
            if ty.endswith("_occur"):
                continue
            if not isinstance(opc_cnt, dict):
                opc_cnt = [(k,v) for k,v in opc_cnt.most_common()]
            else:
                opc_cnt = sorted(list(opc_cnt.items()), key=lambda k:k[1], reverse=True)
            mapped_cnt = 0
            tot_cnt = sum([v for _, v in opc_cnt])
            idmaps[gtype][ty] = len(idmaps[gtype])
            start_id = len(idmaps[gtype])
            for i, (k, v) in enumerate(opc_cnt):
                idmaps[gtype][k] = i + start_id
                mapped_cnt += v
                if isinstance(ths[ty], float):
                    if not freq_mode and mapped_cnt / tot_cnt > ths[ty]:
                        break
                    elif freq_mode and v / num_func < ths[ty]:
                        break
                elif isinstance(ths[ty], int) and i + 1 >= ths[ty]:
                    break
            print("[D] Found: {} mnemonics.".format(len(opc_cnt)))
            print("[D] Num of mnemonics mapped: {}".format(len(idmaps[gtype])-start_id))
        print("[D] Tot Num of mnemonics mapped: {}".format(len(idmaps[gtype])))
    return idmaps


def create_graph_coo_tuple(fva_data):
    NUM_EDGE_TYPE = 4
    NUM_POS_ENC = 8

    nodes, edges = fva_data['nodes'], fva_data['edges']
    nodelist = []
    node_to_idx = {}
    for node in nodes:
        if node not in node_to_idx:
            node_to_idx[node] = len(nodelist)
            nodelist.append(node)
    row = np.empty(len(edges), dtype=np.int64)
    col = np.empty(len(edges), dtype=np.int64)
    data = np.empty(len(edges), dtype=np.int8)
    last_edge, pos_id = (-1, -1), -1
    for idx, edge in enumerate(edges):
        if edge[0] not in node_to_idx:
            node_to_idx[edge[0]] = len(nodelist)
            nodelist.append(edge[0])
        if edge[1] not in node_to_idx:
            node_to_idx[edge[1]] = len(nodelist)
            nodelist.append(edge[1])
        if (edge[0], edge[2]) == last_edge:
            if pos_id + 1 < NUM_POS_ENC:
                pos_id += 1
        else:
            pos_id = 0
            last_edge = (edge[0], edge[2])
        row[idx] = node_to_idx[edge[0]]
        col[idx] = node_to_idx[edge[1]]
        data[idx] = NUM_EDGE_TYPE * pos_id + edge[2]

    return (row, col, data, len(nodelist), len(nodelist)), nodelist


def create_features_matrix(node_list, fva_data, opc_dict, gtype, arch):
    """
    Create the matrix with numerical features.

    Args:
        node_list: list of basic-blocks addresses
        fva_data: dict with features associated to a function
        opc_dict: selected opcodes.

    Return
        np.matrix: Numpy matrix with selected features.
    """
    assert gtype in ['SOG', 'TSCG', 'ISCG', 'ACFG']
    if gtype in ['SOG', 'TSCG']:
        opcs = []
        for node_fva in node_list:
            assert str(node_fva) in fva_data['nverbs']
            node_data = fva_data['nverbs'][str(node_fva)]
            for ty, nopc in process_nverb(gtype, node_data, arch):
                if nopc in opc_dict:
                    opcs.append(opc_dict[nopc])
                else:
                    opcs.append(opc_dict[ty])
        asms = opcs
    elif gtype in ['ISCG', 'ACFG']:
        asms = []
        I_SIZE = 12 if gtype == 'ISCG' else 256
        PADDING = opc_dict['padding']
        assert PADDING == 0
        for node_fva in node_list:
            opcs = np.zeros(I_SIZE, dtype=np.uint16)
            node_data = fva_data['nverbs'].get(str(node_fva), [])
            for i, (ty, nopc) in enumerate(process_nverb(gtype, node_data, arch)):
                if i >= I_SIZE:
                    break
                if nopc in opc_dict:
                    opcs[i] = opc_dict[nopc]
                else:
                    opcs[i] = opc_dict[ty]
            asms.append(opcs)
    return np.array(asms, dtype=np.uint16)


def coo_tuple_to_str(graph_tuple):
    """
    Convert the Numpy matrix in input to a Scipy sparse matrix.

    Args:
        np_mat: a Numpy matrix

    Return
        str: serialized matrix
    """
    row, col, data, n_row, n_col = graph_tuple
    # Custom string serialization
    row_str = ';'.join([str(x) for x in row])
    col_str = ';'.join([str(x) for x in col])
    data_str = ';'.join([str(x) for x in data])
    mat_str = "::".join([row_str, col_str, data_str, str(n_row), str(n_col)])
    return mat_str

def process_one_file(args):
    json_path, opc_dicts, dump_str, dump_pkl = args
    with open(json_path) as f_in:
        jj = json.load(f_in)
    f_json = os.path.basename(json_path)
    arch = f_json.split('-')[0][:-2]
    idb_path = list(jj.keys())[0]
    # print("[D] Processing: {}".format(idb_path))
    str_func_dict, pkl_func_dict = defaultdict(dict), defaultdict(dict)
    j_data = jj[idb_path]
    for key in ['arch', 'failed_functions', 'overrange_functions', 'underrange_functions']:
        if key in j_data:
            del j_data[key]
    # Iterate over each function
    for fva in j_data:
        for gtype in GRAPH_TYPES:
            fva_data = j_data[fva][gtype]
            graph_tuple, nodes = create_graph_coo_tuple(fva_data)
            f_list = create_features_matrix(
                nodes, fva_data, opc_dicts[gtype], gtype, arch)
            if not fva.startswith("0x"):
                fva = hex(int(fva, 10))
            if dump_str:
                str_func_dict[gtype][fva] = {
                    'graph': coo_tuple_to_str(graph_tuple),
                    'opc': f_list
                }
            if dump_pkl:
                pkl_func_dict[gtype][fva] = {
                    'graph': graph_tuple,
                    'opc': f_list
                }
    return idb_path, str_func_dict, pkl_func_dict

def process_one_file_to_shards_safe(args):
    shard_idx, json_path, opc_dicts, dump_str, dump_pkl, shard_dirs = args
    created_paths = []
    try:
        idb_path, str_func_dict, pkl_func_dict = process_one_file(
            (json_path, opc_dicts, dump_str, dump_pkl)
        )
        result = {"ok": True, "json_path": json_path, "shards": {"str": {}, "pkl": {}}}

        if dump_str:
            for gtype, data in str_func_dict.items():
                shard_path = os.path.join(
                    shard_dirs["str"][gtype], f"{shard_idx:06d}.pkl"
                )
                dump_pickle_file((idb_path, data), shard_path)
                created_paths.append(shard_path)
                result["shards"]["str"][gtype] = shard_path

        if dump_pkl:
            for gtype, data in pkl_func_dict.items():
                shard_path = os.path.join(
                    shard_dirs["pkl"][gtype], f"{shard_idx:06d}.pkl"
                )
                dump_pickle_file((idb_path, data), shard_path)
                created_paths.append(shard_path)
                result["shards"]["pkl"][gtype] = shard_path

        return result
    except Exception as exc:
        remove_paths(created_paths)
        return {
            "ok": False,
            "json_path": json_path,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def create_function_shards_parallel(input_folder, output_dir, dataset, freq_mode, opc_dicts, dump_str, dump_pkl):
    shard_dirs = prepare_shard_dirs(output_dir, dataset, freq_mode, dump_str, dump_pkl)
    shard_manifests = {
        "str": {gtype: [] for gtype in GRAPH_TYPES},
        "pkl": {gtype: [] for gtype in GRAPH_TYPES},
    }
    args = []
    for shard_idx, f_json in enumerate(sorted(os.listdir(input_folder))):
        if not f_json.endswith(".json"):
            continue
        json_path = os.path.join(input_folder, f_json)
        args.append((shard_idx, json_path, opc_dicts, dump_str, dump_pkl, shard_dirs))

    failures = []
    worker_count = get_worker_count(len(args))
    chunksize = get_chunksize(len(args), worker_count)
    print(f"[i] create functions dict workers: {worker_count} (chunksize={chunksize})")

    success_count = 0
    with multiprocessing.Pool(processes=worker_count, maxtasksperchild=32) as pool:
        for response in tqdm(
            pool.imap_unordered(process_one_file_to_shards_safe, args, chunksize=chunksize),
            total=len(args),
            desc="create functions dict",
            dynamic_ncols=True,
        ):
            if not response["ok"]:
                failures.append(response)
                continue
            success_count += 1
            if dump_str:
                for gtype, shard_path in response["shards"]["str"].items():
                    shard_manifests["str"][gtype].append(shard_path)
            if dump_pkl:
                for gtype, shard_path in response["shards"]["pkl"].items():
                    shard_manifests["pkl"][gtype].append(shard_path)

    failure_log = write_failure_log(output_dir, dataset, "create_functions_dict", failures)
    if failure_log is not None:
        print(f"[W] Function dict creation failed for {len(failures)} file(s). See {failure_log}")
    if args and success_count == 0:
        cleanup_shard_dirs(shard_dirs)
        raise RuntimeError(
            f"Function dict creation failed for all input files. See {failure_log}"
        )

    for dump_kind in ["str", "pkl"]:
        for gtype in GRAPH_TYPES:
            shard_manifests[dump_kind][gtype].sort()

    print("[D] All shard writers finished.")
    return shard_manifests, shard_dirs


def merge_function_shards(output_dir, dataset, freq_mode, dump_kind, shard_paths_by_gtype):
    for gtype in GRAPH_TYPES:
        shard_paths = shard_paths_by_gtype[gtype]
        if not shard_paths:
            continue

        merged = defaultdict(dict)
        desc = f"merge {gtype.lower()} {dump_kind}"
        for shard_path in tqdm(shard_paths, desc=desc, dynamic_ncols=True):
            idb_path, data = load_pickle_file(shard_path)
            merged[idb_path] = data

        output_path = get_graph_output_path(output_dir, gtype, dataset, freq_mode, dump_kind)
        if dump_kind == "str":
            with open(output_path, "w") as f_out:
                json.dump(merged, f_out)
        else:
            dump_pickle_file(merged, output_path)

        del merged
        gc.collect()


def finalize_function_shards(output_dir, dataset, freq_mode, dump_str, dump_pkl, shard_manifests, shard_dirs):
    if dump_str:
        merge_function_shards(output_dir, dataset, freq_mode, "str", shard_manifests["str"])
    if dump_pkl:
        merge_function_shards(output_dir, dataset, freq_mode, "pkl", shard_manifests["pkl"])
    cleanup_shard_dirs(shard_dirs)


def get_sub_dir(output_dir, gtype, dataset=None):
    if dataset is not None:
        sub_dir = os.path.join(output_dir, f'pcode_{gtype.lower()}', dataset)
    else:
        sub_dir = os.path.join(output_dir, f'pcode_{gtype.lower()}')
    os.makedirs(sub_dir, exist_ok=True)
    return sub_dir


@click.command()
@click.option('-i', '--input-dir', required=True,
              help='A directory that contains JSON-formed SOG/ISCG/TSCG/ACFG. ')
@click.option('--training', required=False, is_flag=True,
              help='In training mode, this script generates a new token mapping. ')
@click.option('--freq-mode', default=False, is_flag=True,
              help='In frequency mode, the number of tokens to map is determined by the frequency of occurrence of the token, rather than by a predefined number/ratio. ')
@click.option('-d', '--opcodes-json',
              default="opcodes_dict.json",
              help='Token mapping result file name. ')
@click.option('-o', '--output-dir', required=True,
              help='Output directory. The output path is formed as output-dir/graph-type/dataset. ')
@click.option('-s', '--dataset', required=True,
              help='The name of dataset. Used as part of the output path. ')
@click.option('-f', '--out_format', default='pkl', required=False,
              help='Output format ("json", "pkl" or "both"). ')
def main(input_dir, training, freq_mode, opcodes_json, output_dir, dataset, out_format):
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if training:
        # Conduct token mapping and save results. 
        opc_dicts = token_mapping_parallel(
            input_dir, output_dir, dataset, freq_mode)
        for gtype in GRAPH_TYPES:
            sub_dir = get_sub_dir(output_dir, gtype)
            output_path = os.path.join(sub_dir, opcodes_json)
            with open(output_path, "w") as f_out:
                json.dump(opc_dicts[gtype], f_out)
    else:
        # Load previous token mapping results. 
        opc_dicts = {}
        for gtype in GRAPH_TYPES:
            sub_dir = get_sub_dir(output_dir, gtype)
            json_path = os.path.join(sub_dir, opcodes_json)
            if not os.path.isfile(json_path):
                print("[!] Error loading {}".format(json_path))
                return
            with open(json_path) as f_in:
                opc_dict = json.load(f_in)
            opc_dicts[gtype] = opc_dict

    # Two 
    dump_str = out_format == "json" or out_format == "both"
    dump_pkl = out_format == "pkl" or out_format == "both"

    shard_manifests, shard_dirs = create_function_shards_parallel(
        input_dir, output_dir, dataset, freq_mode, opc_dicts, dump_str, dump_pkl)
    finalize_function_shards(
        output_dir, dataset, freq_mode, dump_str, dump_pkl, shard_manifests, shard_dirs)


if __name__ == '__main__':
    main()
