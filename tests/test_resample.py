"""Timestamp and quaternion-path checks for Q1 resampling."""

import numpy as np
import pytest

from supermimic.resample import resample_qpos


def test_midpoint_follows_so3_geodesic_and_joint_timestamps():
  qpos = np.zeros((2, 29))
  qpos[0, 3] = 1
  qpos[1, 3:7] = [2**-0.5, 0, 2**-0.5, 0]
  qpos[1, 0] = 1
  qpos[1, 7:] = 2
  result = resample_qpos(qpos, 2, 4)
  assert result.shape == (3, 29)
  np.testing.assert_allclose(result[1, :3], [0.5, 0, 0])
  np.testing.assert_allclose(result[1, 7:], 1)
  np.testing.assert_allclose(
    result[1, 3:7], [np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0], atol=1e-7
  )


def test_antipodal_quaternions_and_noninteger_rate_stay_finite():
  qpos = np.zeros((241, 29))
  qpos[:, 3] = np.where(np.arange(len(qpos)) % 2, -1, 1)
  result = resample_qpos(qpos, 120.0047989, 50)
  assert len(result) == 100
  assert np.isfinite(result).all()
  np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=-1), 1)
  assert (len(result) - 1) / 50 <= (len(qpos) - 1) / 120.0047989


def test_invalid_quaternion_is_rejected():
  with pytest.raises(ValueError, match="unit length"):
    resample_qpos(np.zeros((2, 29)), 50)
