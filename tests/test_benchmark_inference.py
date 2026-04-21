import os
import sys
import unittest
from argparse import Namespace

import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import benchmark_inference as bench


class BenchmarkInferenceTests(unittest.TestCase):

    def test_resolve_infer_csv_path_prefers_override(self):
        config = {
            "testing": {
                "infer_tasks": [["a.csv", "a.pkl"]],
            }
        }
        self.assertEqual(
            bench.resolve_infer_csv_path(config, "custom.csv"),
            "custom.csv",
        )

    def test_resolve_infer_csv_path_reads_first_infer_task(self):
        config = {
            "testing": {
                "infer_tasks": [["a.csv", "a.pkl"], "b.csv"],
            }
        }
        self.assertEqual(bench.resolve_infer_csv_path(config, None), "a.csv")

    def test_summarize_series(self):
        stats = bench.summarize_series([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["count"], 4)
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 4.0)
        self.assertAlmostEqual(stats["p50"], 2.5)
        self.assertAlmostEqual(stats["p95"], 3.85)

    def test_resolve_repo_path(self):
        self.assertTrue(bench.resolve_repo_path("dbs/example.csv").endswith("dbs/example.csv"))
        self.assertEqual(bench.resolve_repo_path("/tmp/example.csv"), "/tmp/example.csv")

    def test_get_builtin_profile(self):
        profile = bench.get_builtin_profile()
        self.assertEqual(profile["profile_name"], bench.DEFAULT_PROFILE_NAME)
        self.assertEqual(profile["num_funcs"], 121164)
        self.assertEqual(tuple(profile["node_counts"].shape), (3,))

    def test_get_builtin_model_config(self):
        config = bench.get_builtin_model_config()
        self.assertEqual(config["config_name"], bench.DEFAULT_MODEL_CONFIG_NAME)
        self.assertEqual(config["encoder"]["name"], "embed")
        self.assertEqual(config["encoder"]["embed"]["n_node_attr"], 521)
        self.assertEqual(config["aggr"]["name"], "msoftv2")
        self.assertEqual(config["batch_size"], 100)

    def test_apply_paper_preset(self):
        args = Namespace(
            paper=True,
            warmup_batches=0,
            repeat=1,
            batches_per_repeat=None,
            batch_sizes=None,
            same_batches_across_repeats=None,
        )
        args = bench.apply_paper_preset(args)
        self.assertEqual(args.warmup_batches, 10)
        self.assertEqual(args.repeat, 10)
        self.assertEqual(args.batches_per_repeat, 20)
        self.assertEqual(args.batch_sizes, [100, 200, 400])
        self.assertTrue(args.same_batches_across_repeats)

    def test_resolve_batch_sizes(self):
        self.assertEqual(bench.resolve_batch_sizes([100, 200], 50), [100, 200])
        self.assertEqual(bench.resolve_batch_sizes(None, 50), [50])

    def test_build_batch_sizes(self):
        self.assertEqual(bench.build_batch_sizes(250, 100), [100, 100, 50])
        self.assertEqual(bench.build_batch_sizes(0, 100), [])

    def test_resolve_dummy_num_functions(self):
        self.assertEqual(bench.resolve_dummy_num_functions(123, None, 5, 100, 999), 123)
        self.assertEqual(bench.resolve_dummy_num_functions(123, 7, 5, 100, 999), 700)
        self.assertEqual(bench.resolve_dummy_num_functions(None, 7, 5, 100, 999), 700)
        self.assertEqual(bench.resolve_dummy_num_functions(None, None, 5, 100, 999), 500)
        self.assertEqual(bench.resolve_dummy_num_functions(None, None, 0, 100, 999), 999)

    def test_build_dummy_batch_shapes(self):
        profile = {
            "num_funcs": 2,
            "node_counts": np.asarray([2, 3], dtype=np.int32),
            "edge_counts": np.asarray([1, 2], dtype=np.int32),
            "feature_tail_shapes": [(), ()],
            "feature_dtype": "int32",
            "max_node_token": 7,
            "max_edge_token": 3,
            "summary": {},
            "features_path": "dummy.pkl",
        }
        config = {
            "encoder": {
                "name": "embed",
                "embed": {
                    "n_node_attr": 16,
                    "n_edge_attr": 4,
                    "n_pos_enc": 2,
                },
            }
        }
        batch = bench.build_dummy_batch(
            sampled_indices=np.asarray([0, 1]),
            profile=profile,
            config=config,
            rng=np.random.default_rng(0),
        )
        node_features, edge_index, edge_attr, graph_idx, batch_size = batch
        self.assertEqual(batch_size, 2)
        self.assertEqual(tuple(node_features.shape), (5,))
        self.assertEqual(tuple(edge_index.shape), (2, 3))
        self.assertEqual(tuple(edge_attr.shape), (3,))
        self.assertEqual(tuple(graph_idx.shape), (5,))

    def test_build_dummy_batch_shapes_with_single_tail_shape(self):
        profile = {
            "num_funcs": 121164,
            "node_counts": np.asarray([2, 3, 4], dtype=np.int32),
            "edge_counts": np.asarray([1, 2, 3], dtype=np.int32),
            "feature_tail_shapes": [()],
            "feature_dtype": "uint16",
            "max_node_token": 7,
            "max_edge_token": 3,
            "summary": {},
            "features_path": "<builtin>",
        }
        config = {
            "encoder": {
                "name": "embed",
                "embed": {
                    "n_node_attr": 16,
                    "n_edge_attr": 4,
                    "n_pos_enc": 2,
                },
            }
        }
        batch = bench.build_dummy_batch(
            sampled_indices=np.asarray([2, 1, 0]),
            profile=profile,
            config=config,
            rng=np.random.default_rng(0),
        )
        node_features, edge_index, edge_attr, graph_idx, batch_size = batch
        self.assertEqual(batch_size, 3)
        self.assertEqual(node_features.dtype, bench.torch.long)
        self.assertEqual(tuple(edge_index.shape), (2, 6))
        self.assertEqual(tuple(edge_attr.shape), (6,))
        self.assertEqual(tuple(graph_idx.shape), (9,))

    def test_aggregate_repeat_results(self):
        repeats = [
            {
                "measured_batches": 2,
                "total_funcs": 20,
                "total_nodes": 200,
                "total_edges": 400,
                "elapsed": {
                    "forward": 0.7,
                    "end_to_end": 1.1,
                },
                "throughput": {
                    "seconds_per_100_funcs_forward": 0.35,
                    "seconds_per_100_funcs_end_to_end": 0.55,
                },
                "peak_memory_bytes": 100,
                "timings": {
                    "load": [0.1, 0.2],
                    "h2d": [0.01, 0.02],
                    "forward": [0.3, 0.4],
                    "end_to_end": [0.5, 0.6],
                },
            },
            {
                "measured_batches": 1,
                "total_funcs": 10,
                "total_nodes": 100,
                "total_edges": 200,
                "elapsed": {
                    "forward": 0.5,
                    "end_to_end": 0.8,
                },
                "throughput": {
                    "seconds_per_100_funcs_forward": 0.5,
                    "seconds_per_100_funcs_end_to_end": 0.8,
                },
                "peak_memory_bytes": 120,
                "timings": {
                    "load": [0.3],
                    "h2d": [0.03],
                    "forward": [0.5],
                    "end_to_end": [0.8],
                },
            },
        ]
        aggregate = bench.aggregate_repeat_results(repeats)
        self.assertEqual(aggregate["repeat_count"], 2)
        self.assertEqual(aggregate["measured_batches"], 3)
        self.assertEqual(aggregate["total_funcs"], 30)
        self.assertEqual(aggregate["peak_memory_bytes_max"], 120)
        self.assertAlmostEqual(
            aggregate["repeat_summary"]["forward_batch"]["mean"],
            ((0.7 / 2) + (0.5 / 1)) / 2,
        )
        self.assertAlmostEqual(
            aggregate["throughput"]["funcs_per_s_end_to_end"],
            30 / (1.1 + 0.8),
        )


if __name__ == "__main__":
    unittest.main()
