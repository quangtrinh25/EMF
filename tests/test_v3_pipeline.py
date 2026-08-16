import unittest

import numpy as np
import torch
import yaml

from datasets.prepare_new_calib import load_config, validate_protocol, workspace_bounds
from datasets.prepare_temporal_v3 import build_source_samples, validate_temporal_protocol
from evaluation.evaluate_v3 import require_test_confirmation
from inference.predict_capsule_pose_v3 import prepare_features
from models.residual_net import ResidualNet
from physics.rotation_repr import (
    euler_deg_to_matrix,
    matrix_to_euler_deg,
    matrix_to_rotation_6d,
    mean_rotation_6d,
    project_matrix_to_so3,
    rotation_6d_to_matrix,
    so3_geodesic_deg,
)
from training.losses_v3 import so3_geodesic_radians_torch


class RotationRepresentationTests(unittest.TestCase):
    def test_signed_euler_rotation_round_trip(self):
        euler = np.asarray([
            [179.44, 5.64, -119.94],
            [179.44, -4.36, -119.94],
            [10.0, 20.0, 30.0],
        ])
        expected = euler_deg_to_matrix(euler)
        values = matrix_to_rotation_6d(expected)
        actual = rotation_6d_to_matrix(values)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        reconstructed = euler_deg_to_matrix(matrix_to_euler_deg(actual))
        np.testing.assert_allclose(reconstructed, expected, atol=1e-6)
        np.testing.assert_allclose(so3_geodesic_deg(values, values), 0.0, atol=1e-5)

    def test_torch_geodesic_is_near_zero_for_same_rotation(self):
        matrix = euler_deg_to_matrix(np.asarray([[179.44, -4.36, -119.94]]))
        values = torch.tensor(matrix_to_rotation_6d(matrix), dtype=torch.float32)
        angle = so3_geodesic_radians_torch(values, values)
        self.assertLess(float(angle.item()), 1e-3)

    def test_opposite_signed_pitch_is_not_scored_as_identical(self):
        negative = matrix_to_rotation_6d(euler_deg_to_matrix(np.asarray([[0.0, -5.0, 0.0]])))
        positive = matrix_to_rotation_6d(euler_deg_to_matrix(np.asarray([[0.0, 5.0, 0.0]])))
        self.assertAlmostEqual(float(so3_geodesic_deg(negative, positive)[0]), 10.0, places=4)

    def test_rotation_ensemble_stays_on_so3(self):
        values = matrix_to_rotation_6d(euler_deg_to_matrix(np.asarray([
            [0.0, -4.0, 0.0],
            [0.0, 4.0, 0.0],
        ])))
        mean = mean_rotation_6d(values[:, None, :])
        matrix = rotation_6d_to_matrix(mean)[0]
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=6)
        self.assertLess(float(so3_geodesic_deg(mean, values[:1])[0]), 4.001)

    def test_projection_fixes_reflection(self):
        reflected = np.diag([1.0, 1.0, -1.0])
        projected = project_matrix_to_so3(reflected)
        self.assertAlmostEqual(float(np.linalg.det(projected)), 1.0, places=6)


class ProtocolTests(unittest.TestCase):
    def test_locked_test_requires_explicit_confirmation(self):
        require_test_confirmation('val', False)
        require_test_confirmation('test', True)
        with self.assertRaises(RuntimeError):
            require_test_confirmation('test', False)

    def test_config_has_exact_100mm_workspace_and_disjoint_folds(self):
        config = load_config('configs/new_calib_v3.yaml')
        validate_protocol(config)
        self.assertEqual(config['workspace']['size_mm'], [100.0, 100.0, 100.0])
        lower, upper = workspace_bounds(config)
        np.testing.assert_allclose(upper - lower, [100.0, 100.0, 100.0])

    def test_temporal_protocol_has_no_session_overlap(self):
        config = load_config('configs/new_calib_v3.yaml')
        roles = validate_temporal_protocol(config)
        self.assertFalse(roles['train'] & roles['dev'])
        self.assertFalse(roles['train'] & roles['test'])
        self.assertFalse(roles['dev'] & roles['test'])

    def test_causal_window_delta_time_and_horizon_alignment(self):
        emf = np.arange(36, dtype=np.float32).reshape(4, 9)
        poses = np.zeros((4, 6), dtype=np.float32)
        poses[:, 0] = np.arange(4)
        source = {
            'family': 'synthetic',
            'capsule': 'test',
            'rotation_regime': 'test',
            'session': 1,
            'source_file': 'synthetic.csv',
            'emf': emf,
            'poses': poses,
            'source_rows': np.arange(4),
            'timestamps_s': np.asarray([0.0, 0.1, 0.3, 0.6]),
        }
        single, temporal = build_source_samples(source, window_size=3, horizon_steps=1, include_dt=True)
        self.assertEqual(len(single), 3)
        self.assertEqual(temporal[0]['emf'].shape, (29,))
        np.testing.assert_allclose(temporal[0]['emf'][:27], np.tile(emf[0], 3))
        np.testing.assert_allclose(temporal[0]['emf'][27:], [0.0, 0.0])
        self.assertEqual(float(temporal[0]['pose_deg'][0]), 1.0)
        self.assertAlmostEqual(float(temporal[1]['emf'][-1]), 0.1)
        self.assertAlmostEqual(float(temporal[1]['target_offset_seconds']), 0.2)

    def test_calibration_aware_features_match_realtime_and_reset_at_segment(self):
        emf = (np.arange(54, dtype=np.float32).reshape(6, 9) + 1.0) * 1e-3
        poses = np.zeros((6, 6), dtype=np.float32)
        source = {
            'family': 'synthetic',
            'capsule': 'test',
            'rotation_regime': 'test',
            'session': 1,
            'source_file': 'synthetic.csv',
            'emf': emf,
            'poses': poses,
            'source_rows': np.arange(6),
            'timestamps_s': None,
            'trajectory_segment_rows': 3,
        }
        _, temporal = build_source_samples(
            source,
            window_size=3,
            horizon_steps=0,
            include_dt=False,
            feature_mode='current_plus_log_ratio',
        )
        metadata = {
            'input_dim': 27,
            'temporal_window': 3,
            'dt_feature_count': 0,
            'temporal_feature_mode': 'current_plus_log_ratio',
        }
        realtime, warmup = prepare_features(
            emf,
            metadata,
            reset_flags=np.asarray([False, False, False, True, False, False]),
            return_warmup=True,
        )
        offline = np.stack([sample['emf'] for sample in temporal])
        np.testing.assert_allclose(realtime, offline, atol=1e-7)
        np.testing.assert_array_equal(warmup, [True, True, False, True, True, False])
        np.testing.assert_allclose(realtime[:, :9], emf)

    def test_unit_row_time_produces_only_causal_delta_steps(self):
        emf = np.arange(36, dtype=np.float32).reshape(4, 9) + 1.0
        metadata = {
            'input_dim': 29,
            'temporal_window': 3,
            'dt_feature_count': 2,
            'temporal_feature_mode': 'raw_window',
            'timestamp_source': 'unit_step_source_row_experiment',
        }
        features, warmup = prepare_features(
            emf, metadata, timestamps_s=np.arange(4, dtype=float), return_warmup=True,
        )
        np.testing.assert_allclose(features[:, -2:], [
            [0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0],
        ])
        np.testing.assert_array_equal(warmup, [True, True, False, False])

    def test_resnet_supports_v3_without_breaking_legacy_default(self):
        legacy = ResidualNet()
        upgraded = ResidualNet(output_dim=9, hidden=32, n_blocks=2)
        self.assertEqual(tuple(legacy(torch.zeros(2, 9)).shape), (2, 6))
        self.assertEqual(tuple(upgraded(torch.zeros(2, 9)).shape), (2, 9))


if __name__ == '__main__':
    unittest.main()
