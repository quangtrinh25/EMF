import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.build_hybrid_v4 import select_stage1_training_arrays

from datasets.prepare_multigeneration_v4 import (
    build_samples,
    fold_id,
    inspect_csv_layout,
    load_source,
    sha256_file,
    training_dt_medians,
    validate_frozen_manifest,
)
from models.localizer_v4 import ResidualLocalizerV4, TemporalGRULocalizerV4
from inference.realtime_predictor_v4 import RealtimePosePredictorV4
from inference.predict_capsule_pose_v3 import prepare_features
from inference.realtime_predictor_hybrid_v4 import fixed_row_features
from evaluation.ensemble_hybrid_v4 import blend_targets
from physics.rotation_repr import pose_deg_to_target_v3
from training.train_v4 import balanced_sample_weights


ROOT = Path(__file__).resolve().parents[1]


def synthetic_source(reset_flags=None, timestamp_shift=0.0):
    emf = (np.arange(54, dtype=np.float32).reshape(6, 9) + 1.0) * 1e-3
    poses = np.zeros((6, 6), dtype=np.float32)
    poses[:, :3] = np.asarray([-99.02, 390.255, 270.88])
    timestamps = np.asarray([0.0, 0.1, 0.2, 10.0, 10.1, 10.2]) + timestamp_shift
    return {
        "generation": "new",
        "family": "continuous",
        "session": 1,
        "source_file": "synthetic.csv",
        "sha256": "synthetic",
        "emf": emf,
        "poses": poses,
        "source_rows": np.arange(6),
        "timestamps_s": timestamps,
        "reset_flags": np.asarray(reset_flags or [True, False, False, True, False, False]),
        "invalid_rows": np.empty(0, dtype=np.int64),
        "timestamp_source": "hardware",
        "trajectory_segment_rows": None,
    }


class V4FeatureTests(unittest.TestCase):
    def test_weighted_rotation6d_blend_preserves_position_and_orientation_endpoints(self):
        left = np.arange(18, dtype=np.float32).reshape(2, 9)
        right = left + 10.0
        output = blend_targets(left, right, 0.2, 0.3, "rotation6d_mean")
        np.testing.assert_allclose(output[:, :3], 0.8 * left[:, :3] + 0.2 * right[:, :3])
        np.testing.assert_allclose(output[:, 3:], 0.7 * left[:, 3:] + 0.3 * right[:, 3:])

    def test_hybrid_builder_can_keep_stage1_synthetic_only(self):
        real = {
            "target": np.asarray([[1.0], [2.0]], dtype=np.float32),
            "generation_index": np.asarray([0, 0], dtype=np.int64),
        }
        synthetic = {
            "target": np.asarray([[3.0], [4.0], [5.0]], dtype=np.float32),
            "generation_index": np.asarray([1, 1, 1], dtype=np.int64),
        }
        pure = select_stage1_training_arrays(real, synthetic, "synthetic_only")
        self.assertEqual(len(pure["target"]), 3)
        np.testing.assert_array_equal(pure["generation_index"], 1)
        mixed = select_stage1_training_arrays(real, synthetic, "mixed")
        self.assertEqual(len(mixed["target"]), 5)

    def test_combined_feature_dimensions_and_identity_adapter(self):
        packed = torch.arange(58, dtype=torch.float32).reshape(2, 29) * 1e-4 + 1e-3
        generation = torch.tensor([0, 1])
        plain = ResidualLocalizerV4(
            window_size=3, feature_mode="combined", generation_count=2,
            use_calibration_adapter=False, hidden=16, n_blocks=1,
        )
        adapted = ResidualLocalizerV4(
            window_size=3, feature_mode="combined", generation_count=2,
            use_calibration_adapter=True, adapter_channel_std=np.ones((2, 9)),
            hidden=16, n_blocks=1,
        )
        plain_features = plain.build_features(packed, generation)
        adapted_features = adapted.build_features(packed, generation)
        self.assertEqual(tuple(plain_features.shape), (2, 47))
        torch.testing.assert_close(plain_features, adapted_features)
        self.assertEqual(adapted.packed_input_dim, 29)

    def test_w5_combined_feature_count_is_85(self):
        model = ResidualLocalizerV4(window_size=5, feature_mode="combined", hidden=16, n_blocks=1)
        features = model.build_features(torch.ones(3, 49), torch.zeros(3, dtype=torch.long))
        self.assertEqual(tuple(features.shape), (3, 85))

    def test_reset_windows_never_cross_and_absolute_time_is_invariant(self):
        source = synthetic_source()
        shifted = synthetic_source(timestamp_shift=1_000_000.0)
        median = training_dt_medians([source])["new"]
        samples = build_samples(source, 0, 3, median, 5.0, 5.0)
        shifted_samples = build_samples(shifted, 0, 3, median, 5.0, 5.0)
        np.testing.assert_allclose(
            np.stack([sample["emf"] for sample in samples]),
            np.stack([sample["emf"] for sample in shifted_samples]),
            atol=1e-6,
        )
        reset_sample = samples[3]["emf"]
        np.testing.assert_allclose(reset_sample[:27], np.tile(source["emf"][3], 3))
        np.testing.assert_allclose(reset_sample[27:], 0.0)

    def test_sealed_source_loader_fails_closed(self):
        descriptor = {
            "locked": False,
            "sealed": True,
            "source_file": "sealed.csv",
        }
        with self.assertRaises(RuntimeError):
            load_source(descriptor)

    def test_sealed_open_receipt_prevents_second_open(self):
        with tempfile.TemporaryDirectory() as temp_raw:
            temp = Path(temp_raw)
            config = temp / "protocol.yaml"
            config.write_text("schema_version: 4\n")
            manifest_path = temp / "freeze_manifest.json"
            manifest_path.write_text(json.dumps({
                "frozen": True,
                "test_opened_at_freeze": False,
                "model_sha256": "model",
                "protocol_config_sha256": sha256_file(config),
                "sealed_source_sha256": "sealed",
            }))
            _, receipt = validate_frozen_manifest(
                manifest_path, config, {"sha256": "sealed"},
            )
            receipt.write_text(json.dumps({"one_time_open_complete": True}))
            with self.assertRaises(RuntimeError):
                validate_frozen_manifest(manifest_path, config, {"sha256": "sealed"})

    def test_headerless_fixed_row_layout_needs_no_timestamp_column(self):
        with tempfile.TemporaryDirectory() as temp_raw:
            path = Path(temp_raw) / "session_01.csv"
            path.write_text(",".join(str(value) for value in range(15)) + "\n")
            has_header, mapping = inspect_csv_layout(
                path, {}, "fixed_row_delta", "auto",
            )
            self.assertFalse(has_header)
            self.assertIsNone(mapping)

    def test_fold_id_is_generation_session_scoped(self):
        self.assertEqual(
            fold_id({"generation": "new", "family": "stream", "session": 2}),
            "new__stream__s2",
        )

    def test_gru_candidate_is_causal_window_model(self):
        model = TemporalGRULocalizerV4(window_size=5, hidden=128)
        model.set_normalization(np.zeros(49), np.ones(49), np.zeros(9), np.ones(9))
        output = model(torch.ones(4, 49), torch.zeros(4, dtype=torch.long))
        self.assertEqual(tuple(output.shape), (4, 9))

    def test_hybrid_realtime_fixed_row_features_match_offline_builder(self):
        emf = (np.arange(36, dtype=np.float32).reshape(4, 9) + 1.0) * 1e-3
        timestamps = np.arange(len(emf), dtype=float)
        reset = np.asarray([True, False, False, False])
        raw_metadata = {
            "temporal_window": 3, "temporal_feature_mode": "raw_window",
            "dt_feature_count": 0, "input_dim": 27,
        }
        rowtime_metadata = {
            **raw_metadata, "dt_feature_count": 2, "input_dim": 29,
        }
        log_metadata = {
            **raw_metadata, "temporal_feature_mode": "current_plus_log_ratio",
        }
        raw_offline = prepare_features(emf, raw_metadata, timestamps, reset)
        rowtime_offline = prepare_features(emf, rowtime_metadata, timestamps, reset)
        log_offline = prepare_features(emf, log_metadata, timestamps, reset)
        buffer = []
        for index, row in enumerate(emf):
            buffer.append(row)
            if len(buffer) > 3:
                buffer.pop(0)
            raw, rowtime, logratio, hybrid = fixed_row_features(buffer)
            np.testing.assert_allclose(raw, raw_offline[index], atol=1e-7)
            np.testing.assert_allclose(rowtime, rowtime_offline[index], atol=1e-7)
            np.testing.assert_allclose(logratio, log_offline[index], atol=1e-7)
            np.testing.assert_allclose(hybrid, rowtime_offline[index], atol=1e-7)


class V4SamplerAndSmokeTests(unittest.TestCase):
    def test_balanced_sampler_weights_are_finite_and_normalized(self):
        protocol = {"workspace": {"center_mm": [0, 0, 0], "size_mm": [100, 100, 100]}}
        target = np.zeros((8, 9), dtype=np.float32)
        target[:, :3] = np.asarray([
            [-49, -49, -49], [-48, -48, -48], [0, 0, 0], [1, 1, 1],
            [49, 49, 49], [48, 48, 48], [10, 10, 10], [11, 11, 11],
        ])
        data = {
            "generation_index": np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
            "session_key": np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"]),
            "target": target,
        }
        weights = balanced_sample_weights(data, protocol)
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_one_epoch_train_and_evaluate_smoke(self):
        with tempfile.TemporaryDirectory() as temp_raw:
            temp = Path(temp_raw)
            data_dir = temp / "data"
            checkpoint_dir = temp / "checkpoint"
            eval_dir = temp / "eval"
            data_dir.mkdir()
            rng = np.random.default_rng(4)
            generation_map = {"old": 0, "new": 1}
            workspace = {"center_mm": [0.0, 0.0, 0.0], "size_mm": [100.0, 100.0, 100.0], "tolerance_mm": 1.0}

            def arrays(count, session):
                pose = np.zeros((count, 6), dtype=np.float32)
                pose[:, :3] = rng.uniform(-20, 20, size=(count, 3))
                pose[:, 3:] = rng.uniform(-5, 5, size=(count, 3))
                return {
                    "emf": rng.uniform(1e-3, 2e-2, size=(count, 29)).astype(np.float32),
                    "target": pose_deg_to_target_v3(pose),
                    "generation_index": np.arange(count, dtype=np.int64) % 2,
                    "session_key": np.asarray([session] * count),
                    "generation": np.asarray(["old" if index % 2 == 0 else "new" for index in range(count)]),
                    "family": np.asarray(["synthetic"] * count),
                    "source_file": np.asarray([f"{session}.csv"] * count),
                    "source_row": np.arange(count),
                    "warmup": np.zeros(count, dtype=bool),
                }

            np.savez_compressed(data_dir / "train.npz", **arrays(24, "train"))
            np.savez_compressed(data_dir / "val.npz", **arrays(8, "val"))
            protocol = {
                "schema_version": 4,
                "candidate": "C1",
                "candidate_config": {
                    "model_type": "resnet", "window_size": 3, "feature_mode": "combined",
                    "calibration_adapter": False, "balanced_sampler": False,
                },
                "packed_input_dim": 29,
                "resnet_feature_dim": 47,
                "window_size": 3,
                "generation_to_index": generation_map,
                "train_dt_median": {"old": 1.0, "new": 0.1},
                "generation_time_mode": {"old": "unit_step_source_row", "new": "fixed_row_delta"},
                "generation_fixed_dt_seconds": {"old": None, "new": None},
                "dt_ratio_clip": 5.0,
                "gap_reset_multiple": 5.0,
                "workspace": workspace,
                "fold": "synthetic",
                "sealed_test_opened": False,
            }
            (data_dir / "protocol.json").write_text(json.dumps(protocol))
            training_config = temp / "training.yaml"
            training_config.write_text(
                "epochs: 1\nbatch_size: 8\nlearning_rate: 0.001\nweight_decay: 0\n"
                "position_loss_weight: 1\norientation_loss_weight: 1\n"
                "adapter_regularization_weight: 0.001\nsession_gain_drift: 0\nnoise_std_mV: 0\n"
                "gradient_clip_norm: 5\nearly_stopping_patience: 2\nseed: 42\ndevice: cpu\n"
                "model:\n  hidden: 16\n  residual_blocks: 1\n  leaky_relu_slope: 0.01\n"
            )
            subprocess.run([
                sys.executable, str(ROOT / "training/train_v4.py"),
                "--data_dir", str(data_dir), "--checkpoint_dir", str(checkpoint_dir),
                "--config", str(training_config), "--device", "cpu", "--epochs", "1",
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            subprocess.run([
                sys.executable, str(ROOT / "evaluation/evaluate_v4.py"),
                "--data_dir", str(data_dir), "--checkpoint_dir", str(checkpoint_dir),
                "--out_dir", str(eval_dir), "--split", "val", "--device", "cpu",
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertTrue((checkpoint_dir / "best.pt").exists())
            with open(eval_dir / "summary.json") as stream:
                summary = json.load(stream)
            self.assertEqual(summary["split"], "val")
            self.assertEqual(int(summary["aggregate"]["num_samples"]), 8)
            predictor = RealtimePosePredictorV4(
                [checkpoint_dir], generation="new", device="cpu", backend="eager", input_unit="V",
            )
            response = predictor.predict(1_000_000_000, True, np.full(9, 0.01))
            self.assertTrue(response["warmup"])
            self.assertEqual(set(response["pose"]), {
                "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg",
            })
            implicit_time_response = predictor.predict(None, False, np.full(9, 0.01))
            self.assertIsNone(implicit_time_response["timestamp_ns"])
            # The public source release intentionally excludes trained
            # checkpoints. Exercise the frozen-baseline bridge when the local
            # deployment bundle is present, while keeping this smoke test
            # self-contained in a clean clone.
            deployment_manifest = (
                ROOT / "checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json"
            )
            if deployment_manifest.exists():
                baseline_dir = temp / "v32_baseline"
                subprocess.run([
                    sys.executable, str(ROOT / "evaluation/evaluate_frozen_v32_on_v4.py"),
                    "--data_dir", str(data_dir), "--out_dir", str(baseline_dir),
                    "--device", "cpu",
                ], cwd=ROOT, check=True, capture_output=True, text=True)
                with open(baseline_dir / "summary.json") as stream:
                    baseline = json.load(stream)
                self.assertFalse(baseline["new_sealed_test_opened"])

    def test_quality_gate_accepts_only_complete_balanced_improvement(self):
        with tempfile.TemporaryDirectory() as temp_raw:
            temp = Path(temp_raw)
            metric_keys = (
                "position_axis_rmse_mm", "position_euclidean_p95_mm",
                "orientation_geodesic_rmse_deg", "orientation_geodesic_p95_deg",
            )
            baseline = {
                "aggregate": {key: 1.0 for key in metric_keys},
                "sessions": {"session": {key: 1.0 for key in metric_keys}},
            }
            candidate = {
                "aggregate": {key: 0.9 for key in metric_keys},
                "sessions": {"session": {key: 0.9 for key in metric_keys}},
            }
            (temp / "baseline.json").write_text(json.dumps(baseline))
            (temp / "candidate.json").write_text(json.dumps(candidate))
            (temp / "latency.json").write_text(json.dumps({"cpu_p95_ms": 2.0, "gpu_p95_ms": 1.0}))
            out = temp / "gate.json"
            subprocess.run([
                sys.executable, str(ROOT / "evaluation/quality_gate_v4.py"),
                "--baseline", str(temp / "baseline.json"),
                "--candidate", str(temp / "candidate.json"),
                "--latency", str(temp / "latency.json"), "--out", str(out),
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            with open(out) as stream:
                self.assertTrue(json.load(stream)["accepted"])
            (temp / "latency.json").write_text(json.dumps({"cpu_p95_ms": 2.0, "gpu_p95_ms": None}))
            subprocess.run([
                sys.executable, str(ROOT / "evaluation/quality_gate_v4.py"),
                "--baseline", str(temp / "baseline.json"),
                "--candidate", str(temp / "candidate.json"),
                "--latency", str(temp / "latency.json"), "--out", str(out),
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            with open(out) as stream:
                self.assertFalse(json.load(stream)["accepted"])


if __name__ == "__main__":
    unittest.main()
