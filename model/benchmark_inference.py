#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#                                                                            #
#  Inference benchmark utility for HermesSim                                 #
#                                                                            #
##############################################################################

import argparse
import copy
import json
import os
import pickle
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils import data

from core import GNNModel, load_config_from_json
from core.build_dataset import build_testing_generator
from core.gnn_model import batch_to


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILE_CACHE_SUFFIX = ".bench_profile.pkl"
DEFAULT_MODEL_CONFIG_NAME = "e00_major_noinfer_sog_builtin"
DEFAULT_BUILTIN_OUTPUT_DIR = os.path.join(REPO_ROOT, ".benchmark_cache", "builtin")
DEFAULT_BUILTIN_FEATURE_PATH = "builtin/pcode_sog/Dataset-1_testing/graph_func_dict_opc_True.pkl"
DEFAULT_PROFILE_NAME = "dataset1_testing_builtin"
DEFAULT_MODEL_CONFIG = {
    "config_name": DEFAULT_MODEL_CONFIG_NAME,
    "ggnn_net": {
        "n_node_feat_dim": 32,
        "n_edge_feat_dim": 8,
        "layer_groups": [1, 1, 1, 1, 1, 1],
        "n_message_net_layers": 3,
        "skip_mode": 0,
        "output_mode": 0,
        "concat_skip": 0,
        "num_query": 4,
        "n_atte_layers": 0,
        "layer_aggr": "add",
        "layer_aggr_kwargs": {},
    },
    "encoder": {
        "name": "embed",
        "embed": {
            "n_node_feat_dim": 32,
            "n_edge_feat_dim": 8,
            "n_node_attr": 521,
            "n_edge_attr": 4,
            "n_pos_enc": 8,
        },
    },
    "aggr": {
        "name": "msoftv2",
        "msoftv2": {
            "num_querys": 6,
            "hidden_channels": 64,
            "n_node_trans": 1,
            "n_agg_trans": 1,
            "q_scale": 1.0,
            "out_method": "lin",
        },
    },
    "used_subgraphs": [1, 2, 3],
    "max_vertices": -1,
    "edge_feature_dim": 8,
    "training": {
        "mode": "batch_pair",
        "loss": "cosine",
        "opt": "circle",
        "gama": 1e7,
        "margin": 0.10,
        "norm_neg_sampling_s": 1.0,
        "graph_vec_regularizer_weight": 1e-8,
        "clip_value": 10.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "num_epochs": 20,
        "batch_size": 80,
        "max_num_nodes": 180000,
        "max_num_edges": 360000,
        "n_sim_funcs": 2,
        "epoh_tolerate": 1e7,
        "clean_cache_after": 1e7,
        "evaluate_after": 1,
        "print_after": 1250,
    },
    "validation": {},
    "testing": {
        "infer_tasks": [],
        "features_testing_path": DEFAULT_BUILTIN_FEATURE_PATH,
    },
    "tunning": {
        "run_test": [],
    },
    "outputdir": DEFAULT_BUILTIN_OUTPUT_DIR,
    "device": "cuda",
    "batch_size": 100,
    "checkpoint_dir": DEFAULT_BUILTIN_OUTPUT_DIR,
    "seed": 11,
}
DEFAULT_DUMMY_PROFILE = {
    "profile_name": DEFAULT_PROFILE_NAME,
    "features_path": "<builtin>",
    "num_funcs": 121164,
    "feature_dtype": "uint16",
    "feature_tail_shapes": [()],
    "max_node_token": 520,
    "max_edge_token": 31,
    "node_counts": np.asarray([309, 2068, 637], dtype=np.int32),
    "edge_counts": np.asarray([544, 3837, 1174], dtype=np.int32),
    "summary": {
        "nodes": {
            "count": 121164,
            "mean": 637.7584596084646,
            "std": 0.0,
            "min": 1.0,
            "p50": 309.0,
            "p95": 2068.0,
            "max": 104543.0,
        },
        "edges": {
            "count": 121164,
            "mean": 1174.1394391073256,
            "std": 0.0,
            "min": 0.0,
            "p50": 544.0,
            "p95": 3836.8499999999913,
            "max": 210118.0,
        },
    },
}


def positive_int(value: str) -> int:
    value_int = int(value)
    if value_int <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value_int


def non_negative_int(value: str) -> int:
    value_int = int(value)
    if value_int < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value_int


def positive_int_list(value: str) -> int:
    return positive_int(value)


def resolve_infer_csv_path(config: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override

    infer_tasks = config.get("testing", {}).get("infer_tasks", [])
    if not infer_tasks:
        raise ValueError(
            "config.json does not contain testing.infer_tasks. "
            "Please pass --csv-path explicitly."
        )

    first_item = infer_tasks[0]
    if isinstance(first_item, (tuple, list)):
        return first_item[0]
    return first_item


def resolve_repo_path(path: Optional[str]) -> Optional[str]:
    if path is None or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def get_builtin_model_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_MODEL_CONFIG)


def apply_paper_preset(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "paper", False):
        return args
    if args.warmup_batches == 0:
        args.warmup_batches = 10
    if args.repeat == 1:
        args.repeat = 10
    if args.batches_per_repeat is None:
        args.batches_per_repeat = 20
    if args.batch_sizes is None:
        args.batch_sizes = [100, 200, 400]
    if args.same_batches_across_repeats is None:
        args.same_batches_across_repeats = True
    return args


def resolve_batch_sizes(
    explicit_batch_sizes: Optional[List[int]],
    default_batch_size: int,
) -> List[int]:
    if explicit_batch_sizes:
        return explicit_batch_sizes
    return [default_batch_size]


def summarize_series(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def format_ms(stats: Dict[str, float]) -> str:
    return (
        f"mean={stats['mean'] * 1e3:.2f} ms, "
        f"p50={stats['p50'] * 1e3:.2f} ms, "
        f"p95={stats['p95'] * 1e3:.2f} ms, "
        f"min={stats['min'] * 1e3:.2f} ms, "
        f"max={stats['max'] * 1e3:.2f} ms"
    )


def format_size_stats(stats: Dict[str, float]) -> str:
    return (
        f"mean={stats['mean']:.2f}, "
        f"p50={stats['p50']:.2f}, "
        f"p95={stats['p95']:.2f}, "
        f"max={stats['max']:.0f}"
    )


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def create_loader(
    generator: Any,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> data.DataLoader:
    loader_kwargs: Dict[str, Any] = dict(
        batch_size=None,
        persistent_workers=num_workers > 0,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return data.DataLoader(generator, **loader_kwargs)


def collect_real_dataset_info(generator: Any) -> Dict[str, Any]:
    raw_gen = generator.gen
    return {
        "mode": "real",
        "num_funcs": int(raw_gen._num_funcs),
        "num_full_batches": int(raw_gen._num_batches_in_epoch),
        "tail_batch_size": int(raw_gen._last_batch_size),
        "batch_size": int(raw_gen._batch_size),
    }


def source_file_meta(path: str) -> Dict[str, int]:
    stat = os.stat(path)
    return {
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_size": int(stat.st_size),
    }


def get_profile_cache_path(features_path: str) -> str:
    return f"{features_path}{PROFILE_CACHE_SUFFIX}"


def feature_to_numpy(feature: Any) -> np.ndarray:
    if isinstance(feature, np.ndarray):
        return feature
    if hasattr(feature, "toarray"):
        return feature.toarray()
    return np.asarray(feature)


def build_feature_profile(features_path: str) -> Dict[str, Any]:
    cache_path = get_profile_cache_path(features_path)
    current_meta = source_file_meta(features_path)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("meta") == current_meta:
            return cached["profile"]

    with open(features_path, "rb") as f:
        feature_dict = pickle.load(f)

    node_counts: List[int] = []
    edge_counts: List[int] = []
    feature_tail_shapes: List[Tuple[int, ...]] = []
    feature_dtype = None
    max_node_token = None
    max_edge_token = None

    for idb_data in feature_dict.values():
        if not isinstance(idb_data, dict):
            continue
        for f_data in idb_data.values():
            if not isinstance(f_data, dict):
                continue

            feature = feature_to_numpy(f_data["opc"])
            graph = f_data["graph"]
            num_nodes = int(graph[3])
            num_edges = int(len(graph[0]))

            node_counts.append(num_nodes)
            edge_counts.append(num_edges)
            feature_tail_shapes.append(tuple(int(x) for x in feature.shape[1:]))

            if feature_dtype is None:
                feature_dtype = str(feature.dtype)

            if feature.size > 0 and feature.dtype.kind in "iu":
                feature_max = int(feature.max())
                max_node_token = (
                    feature_max if max_node_token is None else max(max_node_token, feature_max)
                )

            edge_values = np.asarray(graph[2])
            if edge_values.size > 0:
                edge_max = int(edge_values.max()) - 1
                max_edge_token = (
                    edge_max if max_edge_token is None else max(max_edge_token, edge_max)
                )

    profile = {
        "features_path": features_path,
        "num_funcs": len(node_counts),
        "node_counts": np.asarray(node_counts, dtype=np.int32),
        "edge_counts": np.asarray(edge_counts, dtype=np.int32),
        "feature_tail_shapes": feature_tail_shapes,
        "feature_dtype": feature_dtype or "float32",
        "max_node_token": max_node_token,
        "max_edge_token": max_edge_token,
        "summary": {
            "nodes": summarize_series(node_counts),
            "edges": summarize_series(edge_counts),
        },
    }

    with open(cache_path, "wb") as f:
        pickle.dump({"meta": current_meta, "profile": profile}, f)
    return profile


def get_builtin_profile() -> Dict[str, Any]:
    profile = dict(DEFAULT_DUMMY_PROFILE)
    profile["node_counts"] = DEFAULT_DUMMY_PROFILE["node_counts"].copy()
    profile["edge_counts"] = DEFAULT_DUMMY_PROFILE["edge_counts"].copy()
    profile["feature_tail_shapes"] = list(DEFAULT_DUMMY_PROFILE["feature_tail_shapes"])
    profile["summary"] = json.loads(json.dumps(DEFAULT_DUMMY_PROFILE["summary"]))
    return profile


def build_batch_sizes(num_functions: int, batch_size: int) -> List[int]:
    if num_functions <= 0:
        return []
    full_batches, tail = divmod(num_functions, batch_size)
    batches = [batch_size] * full_batches
    if tail > 0:
        batches.append(tail)
    return batches


def resolve_dummy_num_functions(
    num_functions: Optional[int],
    batches_per_repeat: Optional[int],
    benchmark_batches: int,
    batch_size: int,
    profile_num_funcs: int,
) -> int:
    if batches_per_repeat is not None:
        return batches_per_repeat * batch_size
    if num_functions is not None:
        return num_functions
    if benchmark_batches > 0:
        return benchmark_batches * batch_size
    return profile_num_funcs


def infer_node_token_upper(config: Dict[str, Any], profile: Dict[str, Any]) -> int:
    encoder_name = config["encoder"]["name"]
    if encoder_name == "embed":
        return int(config["encoder"]["embed"]["n_node_attr"])
    if encoder_name == "gru":
        return int(config["encoder"]["gru"]["c"]["embed_size"])
    if encoder_name == "hbmp":
        return int(config["encoder"]["hbmp"]["hbmp_config"]["embed_size"])
    if profile["max_node_token"] is not None:
        return int(profile["max_node_token"]) + 1
    return 1024


def infer_edge_token_upper(config: Dict[str, Any], profile: Dict[str, Any]) -> int:
    encoder_name = config["encoder"]["name"]
    if encoder_name == "embed":
        edge_cfg = config["encoder"]["embed"]
        n_edge_attr = int(edge_cfg["n_edge_attr"])
        n_pos_enc = int(edge_cfg.get("n_pos_enc", 0))
        if n_pos_enc > 0:
            return n_edge_attr * n_pos_enc
        return n_edge_attr
    if profile["max_edge_token"] is not None:
        return int(profile["max_edge_token"]) + 1
    return 4


def make_dummy_node_features(
    num_nodes: int,
    tail_shape: Tuple[int, ...],
    feature_dtype: str,
    node_token_upper: int,
    rng: np.random.Generator,
) -> np.ndarray:
    shape = (num_nodes,) + tail_shape
    dtype = np.dtype(feature_dtype)
    if dtype.kind in "iu":
        return rng.integers(0, node_token_upper, size=shape, dtype=np.int64).astype(dtype)
    return rng.standard_normal(size=shape).astype(dtype)


def build_dummy_batch(
    sampled_indices: np.ndarray,
    profile: Dict[str, Any],
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    node_counts = profile["node_counts"]
    edge_counts = profile["edge_counts"]
    feature_tail_shapes = profile["feature_tail_shapes"]
    feature_dtype = profile["feature_dtype"]

    node_token_upper = infer_node_token_upper(config, profile)
    edge_token_upper = infer_edge_token_upper(config, profile)

    from_idx: List[np.ndarray] = []
    to_idx: List[np.ndarray] = []
    node_features: List[np.ndarray] = []
    graph_idx: List[np.ndarray] = []
    edge_attr: List[np.ndarray] = []

    n_total_nodes = 0
    for graph_id, sample_idx in enumerate(sampled_indices.tolist()):
        num_nodes = int(node_counts[sample_idx])
        num_edges = int(edge_counts[sample_idx])
        if len(feature_tail_shapes) == 1:
            tail_shape = tuple(feature_tail_shapes[0])
        else:
            tail_shape = tuple(feature_tail_shapes[sample_idx])

        features = make_dummy_node_features(
            num_nodes=num_nodes,
            tail_shape=tail_shape,
            feature_dtype=feature_dtype,
            node_token_upper=node_token_upper,
            rng=rng,
        )
        rows = rng.integers(0, num_nodes, size=num_edges, dtype=np.int64)
        cols = rng.integers(0, num_nodes, size=num_edges, dtype=np.int64)
        attrs = rng.integers(0, edge_token_upper, size=num_edges, dtype=np.int64).astype(
            np.int32
        )

        node_features.append(features)
        from_idx.append(rows + n_total_nodes)
        to_idx.append(cols + n_total_nodes)
        edge_attr.append(attrs)
        graph_idx.append(np.full(num_nodes, graph_id, dtype=np.int32))
        n_total_nodes += num_nodes

    node_array = np.concatenate(node_features, axis=0)
    if node_array.dtype.kind in "iu":
        node_tensor = torch.tensor(node_array, dtype=torch.long)
    else:
        node_tensor = torch.tensor(node_array)
    edge_index_tensor = torch.tensor(
        np.array(
            [
                np.concatenate(from_idx, axis=0),
                np.concatenate(to_idx, axis=0),
            ]
        ),
        dtype=torch.long,
    )
    edge_attr_tensor = torch.tensor(np.concatenate(edge_attr, axis=0), dtype=torch.int)
    graph_idx_tensor = torch.tensor(np.concatenate(graph_idx, axis=0), dtype=torch.long)
    return node_tensor, edge_index_tensor, edge_attr_tensor, graph_idx_tensor, len(sampled_indices)


def pin_batch_memory(batch_inputs: Sequence[Any]) -> List[Any]:
    return [
        item.pin_memory() if isinstance(item, torch.Tensor) else item
        for item in batch_inputs
    ]


class DummyBatchIterable:
    def __init__(
        self,
        profile: Dict[str, Any],
        config: Dict[str, Any],
        measured_num_functions: int,
        batch_size: int,
        warmup_batches: int,
        seed: int,
        pin_memory: bool,
        device: torch.device,
    ) -> None:
        self._profile = profile
        self._config = config
        self._measured_num_functions = measured_num_functions
        self._batch_size = batch_size
        self._warmup_batches = warmup_batches
        self._seed = seed
        self._pin_memory = pin_memory and device.type == "cuda"
        self._device = device

        self._warmup_batch_sizes = [batch_size] * warmup_batches
        self._measured_batch_sizes = build_batch_sizes(measured_num_functions, batch_size)
        self._total_num_functions = sum(self._warmup_batch_sizes) + measured_num_functions

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        rng = np.random.default_rng(self._seed)
        pool_size = len(self._profile["node_counts"])
        replace = self._total_num_functions > pool_size
        sampled_indices = rng.choice(
            pool_size,
            size=self._total_num_functions,
            replace=replace,
        )

        offset = 0
        for batch_size in self._warmup_batch_sizes + self._measured_batch_sizes:
            batch_indices = sampled_indices[offset:offset + batch_size]
            batch = build_dummy_batch(batch_indices, self._profile, self._config, rng)
            if self._pin_memory:
                batch = pin_batch_memory(batch)
            yield batch
            offset += batch_size


def prepare_dummy_batches(
    profile: Dict[str, Any],
    config: Dict[str, Any],
    measured_num_functions: int,
    batch_size: int,
    warmup_batches: int,
    seed: int,
    pin_memory: bool,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    return list(
        DummyBatchIterable(
            profile=profile,
            config=config,
            measured_num_functions=measured_num_functions,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            seed=seed,
            pin_memory=pin_memory,
            device=device,
        )
    )


def collect_dummy_dataset_info(
    measured_num_functions: int,
    batch_size: int,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    full_batches, tail = divmod(measured_num_functions, batch_size)
    return {
        "mode": "dummy",
        "num_funcs": measured_num_functions,
        "num_full_batches": full_batches,
        "tail_batch_size": tail,
        "batch_size": batch_size,
        "profile_name": profile.get("profile_name", "custom"),
        "profile_source": profile["features_path"],
        "profile_num_funcs": int(profile["num_funcs"]),
        "profile_summary": profile["summary"],
    }


def build_throughput(
    total_funcs: int,
    total_nodes: int,
    total_edges: int,
    total_end_to_end: float,
    total_forward: float,
) -> Dict[str, float]:
    funcs_per_s_end_to_end = total_funcs / total_end_to_end if total_end_to_end > 0 else 0.0
    funcs_per_s_forward = total_funcs / total_forward if total_forward > 0 else 0.0
    return {
        "funcs_per_s_end_to_end": funcs_per_s_end_to_end,
        "funcs_per_s_forward": funcs_per_s_forward,
        "nodes_per_s_end_to_end": total_nodes / total_end_to_end if total_end_to_end > 0 else 0.0,
        "edges_per_s_end_to_end": total_edges / total_end_to_end if total_end_to_end > 0 else 0.0,
        "seconds_per_100_funcs_end_to_end": 100.0 / funcs_per_s_end_to_end if funcs_per_s_end_to_end > 0 else 0.0,
        "seconds_per_100_funcs_forward": 100.0 / funcs_per_s_forward if funcs_per_s_forward > 0 else 0.0,
    }


def summarize_repeat_level_metrics(repeats: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    per_repeat_forward_batch = [
        rep["elapsed"]["forward"] / rep["measured_batches"]
        for rep in repeats
        if rep["measured_batches"] > 0
    ]
    per_repeat_end_to_end_batch = [
        rep["elapsed"]["end_to_end"] / rep["measured_batches"]
        for rep in repeats
        if rep["measured_batches"] > 0
    ]
    per_repeat_fwd_100 = [
        rep["throughput"]["seconds_per_100_funcs_forward"] for rep in repeats
    ]
    per_repeat_e2e_100 = [
        rep["throughput"]["seconds_per_100_funcs_end_to_end"] for rep in repeats
    ]
    return {
        "forward_batch": summarize_series(per_repeat_forward_batch),
        "end_to_end_batch": summarize_series(per_repeat_end_to_end_batch),
        "seconds_per_100_funcs_forward": summarize_series(per_repeat_fwd_100),
        "seconds_per_100_funcs_end_to_end": summarize_series(per_repeat_e2e_100),
    }


def benchmark_one_repeat(
    gnn_model: GNNModel,
    batch_source: Iterable[Any],
    device: torch.device,
    warmup_batches: int,
    benchmark_batches: int,
    pin_memory: bool,
) -> Dict[str, Any]:
    load_times: List[float] = []
    h2d_times: List[float] = []
    forward_times: List[float] = []
    end_to_end_times: List[float] = []
    batch_funcs: List[int] = []
    batch_nodes: List[int] = []
    batch_edges: List[int] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    batch_iter = iter(batch_source)
    measured_batches = 0
    total_seen_batches = 0

    while True:
        if benchmark_batches > 0 and measured_batches >= benchmark_batches:
            break

        step_start = time.perf_counter()
        try:
            batch_inputs = next(batch_iter)
        except StopIteration:
            break
        after_load = time.perf_counter()

        sync_device(device)
        h2d_start = time.perf_counter()
        batch_inputs = batch_to(
            batch_inputs,
            device,
            non_blocking=pin_memory and device.type == "cuda",
        )
        sync_device(device)
        after_h2d = time.perf_counter()

        _ = gnn_model._embed_one_batch(*batch_inputs)
        sync_device(device)
        after_forward = time.perf_counter()

        is_warmup = total_seen_batches < warmup_batches
        total_seen_batches += 1
        if is_warmup:
            continue

        batch_funcs.append(int(batch_inputs[4]))
        batch_nodes.append(int(batch_inputs[0].shape[0]))
        batch_edges.append(int(batch_inputs[1].shape[1]))
        load_times.append(after_load - step_start)
        h2d_times.append(after_h2d - h2d_start)
        forward_times.append(after_forward - after_h2d)
        end_to_end_times.append(after_forward - step_start)
        measured_batches += 1

    peak_memory_bytes = 0
    if device.type == "cuda":
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))

    total_funcs = int(sum(batch_funcs))
    total_nodes = int(sum(batch_nodes))
    total_edges = int(sum(batch_edges))
    total_end_to_end = float(sum(end_to_end_times))
    total_forward = float(sum(forward_times))

    return {
        "warmup_batches": int(min(warmup_batches, total_seen_batches)),
        "measured_batches": measured_batches,
        "total_seen_batches": total_seen_batches,
        "total_funcs": total_funcs,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "elapsed": {
            "forward": total_forward,
            "end_to_end": total_end_to_end,
        },
        "peak_memory_bytes": peak_memory_bytes,
        "timings": {
            "load": load_times,
            "h2d": h2d_times,
            "forward": forward_times,
            "end_to_end": end_to_end_times,
        },
        "throughput": build_throughput(
            total_funcs=total_funcs,
            total_nodes=total_nodes,
            total_edges=total_edges,
            total_end_to_end=total_end_to_end,
            total_forward=total_forward,
        ),
    }


def aggregate_repeat_results(repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_load = [x for rep in repeats for x in rep["timings"]["load"]]
    all_h2d = [x for rep in repeats for x in rep["timings"]["h2d"]]
    all_forward = [x for rep in repeats for x in rep["timings"]["forward"]]
    all_end_to_end = [x for rep in repeats for x in rep["timings"]["end_to_end"]]

    total_funcs = int(sum(rep["total_funcs"] for rep in repeats))
    total_nodes = int(sum(rep["total_nodes"] for rep in repeats))
    total_edges = int(sum(rep["total_edges"] for rep in repeats))
    total_end_to_end = float(sum(rep["elapsed"]["end_to_end"] for rep in repeats))
    total_forward = float(sum(rep["elapsed"]["forward"] for rep in repeats))

    return {
        "repeat_count": len(repeats),
        "measured_batches": int(sum(rep["measured_batches"] for rep in repeats)),
        "total_funcs": total_funcs,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "elapsed": {
            "forward": total_forward,
            "end_to_end": total_end_to_end,
        },
        "peak_memory_bytes_max": int(
            max((rep["peak_memory_bytes"] for rep in repeats), default=0)
        ),
        "latency": {
            "load": summarize_series(all_load),
            "h2d": summarize_series(all_h2d),
            "forward": summarize_series(all_forward),
            "end_to_end": summarize_series(all_end_to_end),
        },
        "repeat_summary": summarize_repeat_level_metrics(repeats),
        "throughput": build_throughput(
            total_funcs=total_funcs,
            total_nodes=total_nodes,
            total_edges=total_edges,
            total_end_to_end=total_end_to_end,
            total_forward=total_forward,
        ),
    }


def print_report(
    run_dir: str,
    input_source: str,
    dataset_info: Dict[str, Any],
    config: Dict[str, Any],
    checkpoint: Optional[str],
    total_params: int,
    repeats: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
) -> None:
    ggnn_net = config["ggnn_net"]
    print("=== HermesSim Inference Benchmark ===")
    print(f"run_dir: {run_dir}")
    print(f"config_name: {config.get('config_name', '<external>')}")
    print(f"input_mode: {dataset_info['mode']}")
    print(f"input_source: {input_source}")
    print(f"device: {config['device']}")
    print(f"checkpoint: {checkpoint or 'latest in run_dir'}")
    print(
        "model: "
        f"encoder={config['encoder']['name']}, "
        f"aggr={config['aggr']['name']}, "
        f"layer_groups={ggnn_net['layer_groups']}"
    )
    print(f"parameters: {total_params}")
    print(
        "workload: "
        f"funcs={dataset_info['num_funcs']}, "
        f"batch_size={dataset_info['batch_size']}, "
        f"full_batches={dataset_info['num_full_batches']}, "
        f"tail_batch_size={dataset_info['tail_batch_size']}"
    )
    if dataset_info["mode"] == "dummy":
        print(f"profile_name: {dataset_info['profile_name']}")
        print(
            f"profile_source: {dataset_info['profile_source']} "
            f"(funcs={dataset_info['profile_num_funcs']})"
        )
        print(
            "profile(nodes): "
            f"{format_size_stats(dataset_info['profile_summary']['nodes'])}"
        )
        print(
            "profile(edges): "
            f"{format_size_stats(dataset_info['profile_summary']['edges'])}"
        )
    print()

    for idx, rep in enumerate(repeats, start=1):
        peak_mem_gib = rep["peak_memory_bytes"] / (1024 ** 3)
        print(f"[Repeat {idx}]")
        print(
            f"measured_batches={rep['measured_batches']}, "
            f"total_funcs={rep['total_funcs']}, "
            f"e2e_time={rep['elapsed']['end_to_end']:.4f}s, "
            f"fwd_time={rep['elapsed']['forward']:.4f}s, "
            f"peak_mem={peak_mem_gib:.2f} GiB"
        )
        print(
            f"time_per_100_funcs: "
            f"e2e={rep['throughput']['seconds_per_100_funcs_end_to_end']:.4f}s, "
            f"fwd={rep['throughput']['seconds_per_100_funcs_forward']:.4f}s"
        )
        print(
            "latency: "
            f"load[{format_ms(summarize_series(rep['timings']['load']))}] | "
            f"h2d[{format_ms(summarize_series(rep['timings']['h2d']))}] | "
            f"fwd[{format_ms(summarize_series(rep['timings']['forward']))}] | "
            f"e2e[{format_ms(summarize_series(rep['timings']['end_to_end']))}]"
        )
        print()

    peak_mem_gib = aggregate["peak_memory_bytes_max"] / (1024 ** 3)
    print("[Aggregate]")
    print(
        f"repeat_count={aggregate['repeat_count']}, "
        f"measured_batches={aggregate['measured_batches']}, "
        f"total_funcs={aggregate['total_funcs']}, "
        f"e2e_time={aggregate['elapsed']['end_to_end']:.4f}s, "
        f"fwd_time={aggregate['elapsed']['forward']:.4f}s, "
        f"peak_mem_max={peak_mem_gib:.2f} GiB"
    )
    print(
        f"time_per_100_funcs: "
        f"e2e={aggregate['throughput']['seconds_per_100_funcs_end_to_end']:.4f}s, "
        f"fwd={aggregate['throughput']['seconds_per_100_funcs_forward']:.4f}s"
    )
    print(
        f"repeat_mean_std: "
        f"fwd_batch={aggregate['repeat_summary']['forward_batch']['mean'] * 1e3:.2f}±"
        f"{aggregate['repeat_summary']['forward_batch']['std'] * 1e3:.2f} ms, "
        f"fwd_100funcs={aggregate['repeat_summary']['seconds_per_100_funcs_forward']['mean']:.4f}±"
        f"{aggregate['repeat_summary']['seconds_per_100_funcs_forward']['std']:.4f} s"
    )
    print(
        f"load: {format_ms(aggregate['latency']['load'])}\n"
        f"h2d: {format_ms(aggregate['latency']['h2d'])}\n"
        f"fwd: {format_ms(aggregate['latency']['forward'])}\n"
        f"e2e: {format_ms(aggregate['latency']['end_to_end'])}"
    )
    print(
        "throughput: "
        f"funcs/s(e2e)={aggregate['throughput']['funcs_per_s_end_to_end']:.2f}, "
        f"funcs/s(fwd)={aggregate['throughput']['funcs_per_s_forward']:.2f}, "
        f"nodes/s(e2e)={aggregate['throughput']['nodes_per_s_end_to_end']:.2f}, "
        f"edges/s(e2e)={aggregate['throughput']['edges_per_s_end_to_end']:.2f}"
    )


def print_sweep_summary(results: List[Dict[str, Any]]) -> None:
    if len(results) <= 1:
        return
    print()
    print("=== Sweep Summary ===")
    header = (
        "batch_size | repeats | warmup | funcs/repeat | "
        "fwd_ms/repeat(mean) | fwd_ms/repeat(std) | sec/100funcs(fwd mean±std) | peak_mem(GiB)"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        batch_size = item["dataset_info"]["batch_size"]
        aggregate = item["aggregate"]
        repeat_fwd = aggregate["repeat_summary"]["forward_batch"]
        repeat_100 = aggregate["repeat_summary"]["seconds_per_100_funcs_forward"]
        peak_mem_gib = aggregate["peak_memory_bytes_max"] / (1024 ** 3)
        print(
            f"{batch_size:10d} | "
            f"{aggregate['repeat_count']:7d} | "
            f"{item['warmup_batches']:6d} | "
            f"{item['dataset_info']['num_funcs']:12d} | "
            f"{repeat_fwd['mean'] * 1e3:19.2f} | "
            f"{repeat_fwd['std'] * 1e3:18.2f} | "
            f"{repeat_100['mean']:.4f}±{repeat_100['std']:.4f}      | "
            f"{peak_mem_gib:13.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HermesSim inference throughput with real or dummy inputs.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional experiment output directory that contains config.json and checkpoints. If omitted, use the built-in e00_major_noinfer+sog model config.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use paper-oriented defaults: warmup=10, repeat=10, batches_per_repeat=20, batch_sizes=100 200 400.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["real", "dummy"],
        default="dummy",
        help="real reuses the CSV and feature file, dummy samples graph sizes from the dataset profile.",
    )
    parser.add_argument(
        "--profile-mode",
        choices=["builtin", "scan"],
        default="builtin",
        help="builtin uses a baked-in Dataset-1 testing profile; scan rebuilds profile stats from a feature file.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Override the real inference CSV. Defaults to testing.infer_tasks[0] in config.json.",
    )
    parser.add_argument(
        "--profile-features-path",
        default=None,
        help="Override the feature file used to build the dummy size profile.",
    )
    parser.add_argument(
        "--num-functions",
        type=positive_int,
        default=100,
        help="Measured dummy functions per repeat. Warmup batches are added on top.",
    )
    parser.add_argument(
        "--dummy-seed",
        type=int,
        default=None,
        help="Random seed for dummy shape sampling. Defaults to config seed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint path. Defaults to the latest checkpoint in run-dir.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override config device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=None,
        help="Override config batch size for testing.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=positive_int_list,
        default=None,
        help="Run a sweep over multiple batch sizes.",
    )
    parser.add_argument(
        "--batches-per-repeat",
        type=positive_int,
        default=None,
        help="For dummy mode, measured batch count per repeat. If set, num_functions=batch_size*batches_per_repeat.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=non_negative_int,
        default=0,
        help="Number of warmup batches to skip in each repeat.",
    )
    parser.add_argument(
        "--benchmark-batches",
        type=non_negative_int,
        default=50,
        help="For real mode: measured batches per repeat. For dummy mode without --num-functions: measured batch count.",
    )
    parser.add_argument(
        "--repeat",
        type=positive_int,
        default=1,
        help="How many times to rerun the benchmark.",
    )
    parser.add_argument(
        "--num-workers",
        type=non_negative_int,
        default=1,
        help="DataLoader worker count for real mode.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=positive_int,
        default=3,
        help="DataLoader prefetch_factor when num_workers > 0.",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Enable pinned host memory for the DataLoader or dummy CPU tensors.",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned host memory for the DataLoader or dummy CPU tensors.",
    )
    parser.add_argument(
        "--same-batches-across-repeats",
        dest="same_batches_across_repeats",
        action="store_true",
        default=None,
        help="Reuse the exact same dummy workload across repeats to reduce variance.",
    )
    parser.add_argument(
        "--fresh-batches-per-repeat",
        dest="same_batches_across_repeats",
        action="store_false",
        help="Resample a fresh dummy workload for each repeat.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to dump the aggregated benchmark result as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = apply_paper_preset(parse_args())
    if args.same_batches_across_repeats is None:
        args.same_batches_across_repeats = False

    using_external_run = args.run_dir is not None
    if using_external_run:
        run_dir = os.path.abspath(args.run_dir)
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found under {run_dir}")
        config = load_config_from_json(run_dir)
        config["outputdir"] = run_dir
        config["checkpoint_dir"] = run_dir
    else:
        run_dir = "<builtin>"
        config = get_builtin_model_config()
        os.makedirs(config["checkpoint_dir"], exist_ok=True)

    if args.device is not None:
        config["device"] = args.device
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if using_external_run:
        config["testing"]["features_testing_path"] = resolve_repo_path(
            config["testing"]["features_testing_path"]
        )

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    gnn_model = GNNModel(config)
    gnn_model._model_initialize()
    if using_external_run or args.checkpoint is not None:
        gnn_model.restore_model(args.checkpoint)
    gnn_model._model.eval()
    checkpoint_label = args.checkpoint
    if checkpoint_label is None:
        checkpoint_label = "latest in run_dir" if using_external_run else "none (random init)"

    batch_sizes = resolve_batch_sizes(args.batch_sizes, config["batch_size"])
    all_results: List[Dict[str, Any]] = []
    total_params: Optional[int] = None

    for batch_size in batch_sizes:
        config_for_run = copy.deepcopy(config)
        config_for_run["batch_size"] = batch_size
        repeats: List[Dict[str, Any]] = []
        input_source = ""
        dataset_info: Dict[str, Any]

        if args.input_mode == "real":
            if not using_external_run:
                raise ValueError(
                    "Built-in config mode only supports dummy benchmarking. "
                    "Use --run-dir for real-data benchmarking."
                )
            input_source = resolve_repo_path(resolve_infer_csv_path(config_for_run, args.csv_path))
            first_generator = build_testing_generator(config_for_run, input_source)
            dataset_info = collect_real_dataset_info(first_generator)
            for repeat_idx in range(args.repeat):
                if repeat_idx == 0:
                    generator = first_generator
                else:
                    generator = build_testing_generator(config_for_run, input_source)
                batch_source = create_loader(
                    generator,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                    pin_memory=args.pin_memory,
                )
                result = benchmark_one_repeat(
                    gnn_model=gnn_model,
                    batch_source=batch_source,
                    device=device,
                    warmup_batches=args.warmup_batches,
                    benchmark_batches=args.benchmark_batches,
                    pin_memory=args.pin_memory,
                )
                if result["measured_batches"] == 0:
                    raise RuntimeError(
                        "No measured batches were collected. "
                        "Please reduce --warmup-batches or use a larger CSV."
                    )
                repeats.append(result)
        else:
            if args.profile_mode == "builtin":
                profile = get_builtin_profile()
                features_path = profile["features_path"]
            else:
                features_path = resolve_repo_path(
                    args.profile_features_path or config_for_run["testing"]["features_testing_path"]
                )
                profile = build_feature_profile(features_path)
            measured_num_functions = resolve_dummy_num_functions(
                num_functions=args.num_functions,
                batches_per_repeat=args.batches_per_repeat,
                benchmark_batches=args.benchmark_batches,
                batch_size=config_for_run["batch_size"],
                profile_num_funcs=profile["num_funcs"],
            )
            input_source = features_path
            dataset_info = collect_dummy_dataset_info(
                measured_num_functions=measured_num_functions,
                batch_size=config_for_run["batch_size"],
                profile=profile,
            )
            seed_base = config_for_run.get("seed", 0) if args.dummy_seed is None else args.dummy_seed
            prepared_batches = None
            if args.same_batches_across_repeats:
                prepared_batches = prepare_dummy_batches(
                    profile=profile,
                    config=config_for_run,
                    measured_num_functions=measured_num_functions,
                    batch_size=config_for_run["batch_size"],
                    warmup_batches=args.warmup_batches,
                    seed=seed_base,
                    pin_memory=args.pin_memory,
                    device=device,
                )
            for repeat_idx in range(args.repeat):
                if prepared_batches is not None:
                    batch_source = prepared_batches
                else:
                    batch_source = prepare_dummy_batches(
                        profile=profile,
                        config=config_for_run,
                        measured_num_functions=measured_num_functions,
                        batch_size=config_for_run["batch_size"],
                        warmup_batches=args.warmup_batches,
                        seed=seed_base + repeat_idx,
                        pin_memory=args.pin_memory,
                        device=device,
                    )
                result = benchmark_one_repeat(
                    gnn_model=gnn_model,
                    batch_source=batch_source,
                    device=device,
                    warmup_batches=args.warmup_batches,
                    benchmark_batches=0,
                    pin_memory=args.pin_memory,
                )
                if result["measured_batches"] == 0:
                    raise RuntimeError(
                        "No measured dummy batches were collected. "
                        "Please reduce --warmup-batches or increase --num-functions."
                    )
                repeats.append(result)

        if total_params is None:
            total_params = sum(p.numel() for p in gnn_model._model.parameters())
        aggregate = aggregate_repeat_results(repeats)
        print_report(
            run_dir=run_dir,
            input_source=input_source,
            dataset_info=dataset_info,
            config=config_for_run,
            checkpoint=checkpoint_label,
            total_params=total_params,
            repeats=repeats,
            aggregate=aggregate,
        )
        all_results.append(
            {
                "dataset_info": dataset_info,
                "aggregate": aggregate,
                "repeats": repeats,
                "warmup_batches": args.warmup_batches,
                "input_source": input_source,
                "config": {
                    "batch_size": config_for_run["batch_size"],
                    "device": config_for_run["device"],
                },
            }
        )

    assert total_params is not None
    print_sweep_summary(all_results)

    if args.save_json is not None:
        payload = {
            "run_dir": run_dir,
            "input_mode": args.input_mode,
            "paper": args.paper,
            "batch_sizes": batch_sizes,
            "config_summary": {
                "device": config["device"],
                "encoder": config["encoder"]["name"],
                "aggr": config["aggr"]["name"],
                "layer_groups": config["ggnn_net"]["layer_groups"],
                "parameters": total_params,
            },
            "runs": all_results,
        }
        save_path = os.path.abspath(args.save_json)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON saved to {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
