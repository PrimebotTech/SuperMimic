"""Resumable, read-only CPU screening of named Q1 motion archives.

This module deliberately imports neither Torch nor the training environment.
Accepted means passing this documented screen, not proven dynamic feasibility.
"""

import argparse
import fcntl
import hashlib
import io
import json
import multiprocessing
import os
import re
import signal
import sqlite3
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import mujoco
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Status = Literal["accepted", "rejected", "review"]
API = cast(Any, mujoco)
SCHEMA_VERSION = 2
_geometry: "CollisionGeometry | None" = None
_config: "FilterConfig | None" = None
_root: Path | None = None
_labels: dict[str, Any] = {}
_stop = False


@dataclass(frozen=True)
class FilterConfig:
  """Q1-sized screening thresholds in meters, seconds and radians."""

  fps: float = 50.0
  sample_hz: float = 10.0
  collision_review_m: float = 0.005
  collision_reject_m: float = 0.020
  collision_excess_reject_m: float = 0.015
  ground_review_m: float = 0.015
  ground_reject_m: float = 0.040
  unsupported_clearance_m: float = 0.030
  stationary_clearance_m: float = 0.050
  stationary_seconds: float = 0.4
  long_flight_seconds: float = 1.0
  flight_acceleration_tolerance: float = 4.0
  flight_fit_rmse_m: float = 0.020
  joint_review_rad: float = 0.010
  joint_reject_rad: float = 0.050
  allow_unlabeled: bool = False

  def validate(self) -> None:
    values = asdict(self)
    if any(
      not np.isfinite(value) or value <= 0
      for key, value in values.items()
      if key != "allow_unlabeled"
    ):
      raise ValueError("Screening thresholds must be finite and positive")
    if self.sample_hz > self.fps:
      raise ValueError("sample_hz cannot exceed the archive FPS")
    if self.collision_review_m >= self.collision_reject_m:
      raise ValueError("Collision review threshold must precede rejection")


@dataclass(frozen=True)
class CollisionSample:
  """Separate geometric maxima from an actual severe-pair witness."""

  absolute_depth: float = 0.0
  maximum_excess: float = 0.0
  worst_pair: int | None = None
  worst_pair_depth: float = 0.0
  severe_pair: int | None = None
  severe_depth: float = 0.0
  severe_excess: float = 0.0


def sequence_id(relative: str) -> str:
  path = Path(relative)
  return str(path.parent / re.sub(r"__clip\d+$", "", path.stem))


def words(text: str) -> list[str]:
  text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
  return re.findall(r"[a-z]+", text.lower())


def classify_labels(
  relative: str, metadata: dict[str, str], labels: dict[str, Any]
) -> tuple[Status, list[dict[str, Any]]]:
  """Reject explicit interactions; keep ambiguous/opaque labels for review."""
  annotated = labels.get(relative, labels.get(sequence_id(relative)))
  if annotated is not None:
    kind = annotated.get("classification")
    evidence = annotated.get("evidence", "")
    if kind not in ("non_interaction", "object_interaction", "review") or not evidence:
      raise ValueError("Label annotations require classification and evidence")
    status: Status = "review"
    if kind == "non_interaction":
      status = "accepted"
    elif kind == "object_interaction":
      status = "rejected"
    return status, [{"code": "annotation", "evidence": evidence, "kind": kind}]
  corpus = Path(relative).parts[0].lower()
  text = " ".join((Path(relative).stem, metadata.get("motion_key", "")))
  tokens = set(words(text))
  if corpus == "omomo":
    return "rejected", [
      {
        "code": "object_interaction_corpus",
        "evidence": "OMOMO consists of human-object manipulation sequences",
        "matched_source": corpus,
      }
    ]
  compact = re.sub(r"[^a-z]+", " ", text.lower())
  objects = {
    "smalltable",
    "largetable",
    "smallbox",
    "largebox",
    "woodchair",
    "whitechair",
    "plasticbox",
    "suitcase",
    "trashcan",
    "floorlamp",
    "clothesstand",
    "vacuum",
  }
  explicit = sorted(objects & tokens)
  phrases = (
    "carry box",
    "carrying box",
    "lift box",
    "lifting box",
    "push cart",
    "pull cart",
    "climb ladder",
    "climbing ladder",
    "ride bike",
    "riding bike",
    "sit chair",
    "sitting chair",
    "dribbling basketball",
  )
  matched = [
    phrase
    for phrase in phrases
    if re.search(r"(?:^| )" + re.escape(phrase) + r"(?: |$)", compact)
  ]
  interaction_verbs = tokens & {
    "carry",
    "carrying",
    "lift",
    "lifting",
    "push",
    "pushing",
    "pull",
    "pulling",
    "drag",
    "dragging",
    "hold",
    "holding",
    "sit",
    "sitting",
    "ride",
    "riding",
    "climb",
    "climbing",
    "throw",
    "throwing",
  }
  if matched or (explicit and interaction_verbs):
    return "rejected", [
      {
        "code": "explicit_object_interaction_label",
        "matched_tokens": explicit,
        "matched_phrases": matched,
        "matched_verbs": sorted(interaction_verbs),
        "label": text,
      }
    ]
  if explicit:
    return "review", [
      {
        "code": "prop_mentioned_without_action_evidence",
        "matched_tokens": explicit,
        "label": text,
      }
    ]
  ambiguous = tokens & {
    "carry",
    "carrying",
    "hold",
    "holding",
    "push",
    "pull",
    "chair",
    "table",
    "box",
    "ladder",
    "sit",
    "sitting",
    "climb",
    "climbing",
    "reach",
    "reaching",
    "object",
    "interact",
    "interaction",
    "basketball",
    "guitar",
    "door",
    "stairs",
    "stair",
    "staircase",
  }
  if ambiguous:
    return "review", [
      {
        "code": "ambiguous_interaction_label",
        "matched_tokens": sorted(ambiguous),
        "label": text,
      }
    ]
  safe_actions = {
    "walk",
    "walking",
    "run",
    "running",
    "jog",
    "jogging",
    "sprint",
    "jump",
    "jumping",
    "land",
    "hop",
    "idle",
    "stand",
    "standing",
    "dance",
    "dancing",
    "turn",
    "sidestep",
    "locomotion",
    "gesture",
    "gestures",
    "conversation",
    "waving",
    "wave",
    "punch",
    "kick",
    "boxing",
    "fight",
    "martial",
    "stretch",
    "stretching",
    "squat",
    "squats",
    "crawl",
    "crawling",
    "kneel",
    "balance",
    "fall",
    "recovery",
    "stepping",
    "limp",
    "tiptoe",
    "toes",
    "backwards",
    "forward",
    "sideways",
    "crouch",
    "crouching",
    "getup",
    "get",
    "skip",
    "skipping",
  }
  if corpus == "100style" or tokens & safe_actions:
    return "accepted", [{"code": "descriptive_noninteraction_label", "label": text}]
  return "review", [
    {
      "code": "unlabeled_semantics",
      "evidence": "No action annotation identifies whether an object is required",
      "label": text,
    }
  ]


class CollisionGeometry:
  """Measure actual XML collision proxies, independent of contact masks."""

  def __init__(self, xml: Path):
    self.model: Any = mujoco.MjModel.from_xml_path(str(xml))
    self.data: Any = mujoco.MjData(self.model)
    m = self.model
    if (m.nq, m.nv, m.nu) != (29, 28, 22) or m.body(1).name != "pelvis":
      raise ValueError("Expected the Q1 V2 free base and 22 actuators")
    self.joint_names = [m.joint(j).name for j in range(1, 23)]
    # Disabled visual meshes are intentionally excluded. Contact masks do not
    # suppress signed-distance queries on the remaining physical proxies.
    self.geoms = np.asarray(
      [
        i
        for i in range(m.ngeom)
        if m.geom_bodyid[i] and (m.geom_contype[i] or m.geom_conaffinity[i])
      ],
      dtype=np.int32,
    )
    feet = {m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")}
    self.foot_indices = np.flatnonzero(np.isin(m.geom_bodyid[self.geoms], list(feet)))
    if not len(self.foot_indices):
      raise ValueError("Q1 collision feet are missing")
    self.kinds = m.geom_type[self.geoms]
    self.sizes = m.geom_size[self.geoms]
    self.mesh_vertices: dict[int, Array] = {}
    for offset, geom in enumerate(self.geoms):
      if m.geom_type[geom] == API.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[geom]
        start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        self.mesh_vertices[offset] = m.mesh_vert[start : start + count].astype(float)
    floors = [i for i in range(m.ngeom) if m.geom_type[i] == API.mjtGeom.mjGEOM_PLANE]
    if len(floors) != 1 or m.geom_bodyid[floors[0]] != 0:
      raise ValueError("Screening requires one world-fixed horizontal plane")
    self.pose(m.qpos0)
    plane = floors[0]
    if not np.allclose(self.data.geom_xmat[plane].reshape(3, 3)[2], [0, 0, 1]):
      raise ValueError("Screening requires a horizontal ground plane")
    self.floor_z = float(self.data.geom_xpos[plane, 2])
    self.gravity_z = float(m.opt.gravity[2])
    pairs = []
    excluded: Counter[str] = Counter()
    adjacent_clusters = set()
    for body in range(1, m.nbody):
      parent = int(m.body_parentid[body])
      first, second = int(m.body_weldid[body]), int(m.body_weldid[parent])
      if first != second:
        adjacent_clusters.add(tuple(sorted((first, second))))
    for index, first in enumerate(self.geoms):
      b1 = int(m.geom_bodyid[first])
      for second in self.geoms[index + 1 :]:
        b2 = int(m.geom_bodyid[second])
        if b1 == b2 or m.body_weldid[b1] == m.body_weldid[b2]:
          excluded["same_or_welded_body"] += 1
        elif (
          tuple(sorted((int(m.body_weldid[b1]), int(m.body_weldid[b2]))))
          in adjacent_clusters
        ):
          excluded["adjacent_welded_clusters"] += 1
        else:
          pairs.append((int(first), int(second)))
    self.pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    self.bound_sum = m.geom_rbound[self.pairs].sum(1)
    self.baseline = np.zeros(len(self.pairs))
    # Only the XML's specified neutral pose establishes proxy-overlap baseline.
    # No dataset motion is allowed to normalize away its own collision.
    for index in self.candidates():
      a, b = self.pairs[index]
      self.baseline[index] = max(
        0.0, -API.mj_geomDistance(m, self.data, int(a), int(b), 0.1, None)
      )
    self.excluded = dict(excluded)
    binary = np.empty(API.mj_sizeModel(m), dtype=np.uint8)
    API.mj_saveModel(m, None, binary)
    self.model_sha256 = hashlib.sha256(binary.tobytes()).hexdigest()

  def pose(self, qpos: Array) -> None:
    self.data.qpos[:] = qpos
    API.mj_kinematics(self.model, self.data)

  def candidates(self) -> NDArray[np.int64]:
    positions = self.data.geom_xpos
    delta = positions[self.pairs[:, 0]] - positions[self.pairs[:, 1]]
    # Bounding spheres are conservative; a skipped pair cannot penetrate.
    return np.flatnonzero(np.einsum("ij,ij->i", delta, delta) <= self.bound_sum**2)

  def lower_bounds(self) -> Array:
    rows = self.data.geom_xmat[self.geoms].reshape(-1, 3, 3)[:, 2, :]
    support = np.zeros(len(self.geoms))
    for kind in np.unique(self.kinds):
      mask = self.kinds == kind
      row, size = rows[mask], self.sizes[mask]
      if kind == API.mjtGeom.mjGEOM_SPHERE:
        support[mask] = size[:, 0]
      elif kind == API.mjtGeom.mjGEOM_BOX:
        support[mask] = (np.abs(row) * size).sum(1)
      elif kind == API.mjtGeom.mjGEOM_CAPSULE:
        support[mask] = size[:, 0] + size[:, 1] * np.abs(row[:, 2])
      elif kind == API.mjtGeom.mjGEOM_CYLINDER:
        support[mask] = size[:, 0] * np.linalg.norm(row[:, :2], axis=1)
        support[mask] += size[:, 1] * np.abs(row[:, 2])
      elif kind == API.mjtGeom.mjGEOM_ELLIPSOID:
        support[mask] = np.linalg.norm(row * size, axis=1)
      elif kind != API.mjtGeom.mjGEOM_MESH:
        raise ValueError(f"Unsupported collision proxy type: {kind}")
    lower = self.data.geom_xpos[self.geoms, 2] - support - self.floor_z
    for index, vertices in self.mesh_vertices.items():
      lower[index] += float((vertices @ rows[index]).min())
    return lower

  def penetration(
    self, reject_depth: float = 0.02, reject_excess: float = 0.015
  ) -> CollisionSample:
    maximum, excess, worst_depth, severe_depth, severe_excess = 0.0, 0.0, 0.0, 0.0, 0.0
    worst, severe_pair = None, None
    for index in self.candidates():
      a, b = self.pairs[index]
      depth = max(
        0.0, -API.mj_geomDistance(self.model, self.data, int(a), int(b), 0.1, None)
      )
      additional = max(0.0, depth - float(self.baseline[index]))
      maximum = max(maximum, depth)
      if additional > excess:
        worst_depth, excess, worst = depth, additional, int(index)
      if depth >= reject_depth and additional >= reject_excess:
        if additional > severe_excess:
          severe_pair, severe_depth, severe_excess = int(index), depth, additional
    return CollisionSample(
      maximum, excess, worst, worst_depth, severe_pair, severe_depth, severe_excess
    )

  def describe_pair(self, index: int) -> dict[str, Any]:
    m = self.model
    geoms = self.pairs[index].tolist()
    return {
      "geom_ids": geoms,
      "bodies": [m.body(int(m.geom_bodyid[g])).name for g in geoms],
      "neutral_overlap_m": float(self.baseline[index]),
    }

  def calibration(self) -> dict[str, Any]:
    self.pose(self.model.qpos0)
    lower = self.lower_bounds()
    return {
      "model_sha256": self.model_sha256,
      "physical_geom_count": len(self.geoms),
      "tested_pair_count": len(self.pairs),
      "excluded_pairs": self.excluded,
      "neutral_foot_clearance_m": float(lower[self.foot_indices].min()),
      "neutral_body_clearance_m": float(lower.min()),
      "floor_z": self.floor_z,
      "gravity_z": self.gravity_z,
      "neutral_overlaps": [
        {**self.describe_pair(i), "depth_m": float(depth)}
        for i, depth in enumerate(self.baseline)
        if depth > 0.001
      ],
      "distance_queries_ignore_contact_masks": True,
      "visual_meshes_excluded": True,
    }


def spans(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
  edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
  return [
    (int(a), int(b))
    for a, b in zip(
      np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
    )
  ]


def read_motion(
  path: Path, geometry: CollisionGeometry, fps: float
) -> tuple[Array, dict[str, str], str]:
  raw = path.read_bytes()
  with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
    metadata = {
      key: str(archive[key].item())
      for key in (
        "robot_name",
        "root_quat_order",
        "motion_key",
        "source_file",
        "source_qpos_sha256",
        "import_method",
        "resample_method",
      )
      if key in archive
    }
    if metadata.get("robot_name") != "q1_v2":
      raise ValueError("robot_name must be q1_v2")
    if metadata.get("root_quat_order") != "wxyz":
      raise ValueError("root_quat_order must be wxyz")
    if not np.isclose(float(archive["fps"].item()), fps):
      raise ValueError(f"Expected {fps:g} FPS; resample explicitly first")
    names = archive["robot_joint_names"].tolist()
    if len(names) != 22 or set(names) != set(geometry.joint_names):
      raise ValueError("Missing, duplicated or incompatible Q1 joint names")
    qpos = archive["qpos"].astype(np.float64)
  if qpos.ndim != 2 or qpos.shape[1] != 29 or len(qpos) < 20:
    raise ValueError("Expected at least 20 frames of 29-dimensional qpos")
  if not np.isfinite(qpos).all():
    raise ValueError("Nonfinite qpos")
  norms = np.linalg.norm(qpos[:, 3:7], axis=1)
  if np.any(np.abs(norms - 1) > 0.01):
    raise ValueError("Root quaternion is not unit length")
  qpos[:, 3:7] /= norms[:, None]
  qpos[:, 7:] = qpos[:, [7 + names.index(n) for n in geometry.joint_names]]
  return qpos, metadata, hashlib.sha256(raw).hexdigest()


def screen_motion(
  path: Path,
  root: Path,
  geometry: CollisionGeometry,
  config: FilterConfig,
  labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
  relative = str(path.relative_to(root))
  stat = path.stat()
  row: dict[str, Any] = {
    "path": relative,
    "sequence_id": sequence_id(relative),
    "file_size": stat.st_size,
    "mtime_ns": stat.st_mtime_ns,
    "status": "accepted",
    "reasons": [],
    "measurements": {},
  }
  reasons: list[dict[str, Any]] = row["reasons"]

  def reason(status: Status, code: str, **evidence: Any) -> None:
    reasons.append({"code": code, "severity": status, **evidence})
    order = {"accepted": 0, "review": 1, "rejected": 2}
    if order[status] > order[row["status"]]:
      row["status"] = status

  try:
    qpos, metadata, file_hash = read_motion(path, geometry, config.fps)
  except (ValueError, KeyError, OSError) as error:
    reason("rejected", "invalid_archive_contract", error=str(error))
    return row
  row.update({"file_sha256": file_hash, "metadata": metadata, "frames": len(qpos)})
  label_status, label_reasons = classify_labels(relative, metadata, labels or {})
  if config.allow_unlabeled and all(
    r["code"] == "unlabeled_semantics" for r in label_reasons
  ):
    label_status = "accepted"
  for item in label_reasons:
    reason(label_status, item["code"], **{k: v for k, v in item.items() if k != "code"})
  m = geometry.model
  limits = m.jnt_range[1:23]
  limited = m.jnt_limited[1:23].astype(bool)
  outside = np.maximum(limits[:, 0] - qpos[:, 7:], qpos[:, 7:] - limits[:, 1])
  outside = np.maximum(outside, 0) * limited
  violation = float(outside.max())
  if violation > config.joint_review_rad:
    frame, joint = np.unravel_index(outside.argmax(), outside.shape)
    reason(
      "rejected" if violation > config.joint_reject_rad else "review",
      "joint_limit_violation",
      max_rad=violation,
      frame=int(frame),
      joint=geometry.joint_names[joint],
      value_rad=float(qpos[frame, 7 + joint]),
      range_rad=limits[joint].tolist(),
    )
  translation = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1)
  joint_delta = np.abs(np.diff(qpos[:, 7:], axis=0))
  rotation = 2 * np.arccos(
    np.clip(np.abs((qpos[1:, 3:7] * qpos[:-1, 3:7]).sum(1)), 0, 1)
  )
  for name, values, threshold in (
    ("root_translation_discontinuity", translation, 0.30),
    ("root_rotation_discontinuity", rotation, 0.60),
    ("joint_discontinuity", joint_delta.max(1), 0.80),
  ):
    if values.max() > threshold:
      reason(
        "rejected",
        name,
        maximum=float(values.max()),
        frame=int(values.argmax() + 1),
        threshold=threshold,
      )
  stride = max(1, int(round(config.fps / config.sample_hz)))
  indices = np.unique(np.append(np.arange(0, len(qpos), stride), len(qpos) - 1))
  times = indices / config.fps
  body_lower, foot_lower, com_z, depths, excesses = [], [], [], [], []
  worst_pair, worst_frame, worst_excess = None, None, 0.0
  worst_depth = 0.0
  severe_witness: dict[str, Any] | None = None
  for frame in indices:
    geometry.pose(qpos[frame])
    lower = geometry.lower_bounds()
    body_lower.append(float(lower.min()))
    foot_lower.append(float(lower[geometry.foot_indices].min()))
    com_z.append(
      float(np.dot(m.body_mass, geometry.data.xipos[:, 2]) / m.body_mass.sum())
    )
    collision = geometry.penetration(
      config.collision_reject_m, config.collision_excess_reject_m
    )
    depths.append(collision.absolute_depth)
    excesses.append(collision.maximum_excess)
    if collision.maximum_excess > worst_excess:
      worst_pair, worst_frame = collision.worst_pair, int(frame)
      worst_excess, worst_depth = collision.maximum_excess, collision.worst_pair_depth
    if collision.severe_pair is not None and (
      severe_witness is None or collision.severe_excess > severe_witness["excess_m"]
    ):
      severe_witness = {
        "frame": int(frame),
        "depth_m": collision.severe_depth,
        "excess_m": collision.severe_excess,
        **geometry.describe_pair(collision.severe_pair),
      }
  lower_array = np.asarray(body_lower)
  com = np.asarray(com_z)
  sample_depth = np.asarray(depths)
  sample_excess = np.asarray(excesses)
  measurements = row["measurements"]
  measurements.update(
    {
      "sampled_frames": len(indices),
      "geometry_stride_frames": stride,
      "actual_sample_hz": config.fps / stride,
      "sampled_body_clearance_min_m": float(lower_array.min()),
      "sampled_foot_clearance_min_m": float(min(foot_lower)),
      "sampled_foot_clearance_median_m": float(np.median(foot_lower)),
      "sampled_max_self_penetration_m": float(sample_depth.max()),
      "sampled_max_new_self_penetration_m": float(sample_excess.max()),
      "full_frame_max_joint_limit_violation_rad": violation,
      "full_frame_max_joint_speed_rad_s": float(joint_delta.max() * config.fps),
    }
  )
  if severe_witness is not None:
    reason(
      "rejected",
      "self_collision",
      **severe_witness,
      affected_samples=int((sample_excess > config.collision_review_m).sum()),
    )
  elif worst_pair is not None and worst_excess > config.collision_review_m:
    reason(
      "review",
      "self_collision",
      frame=worst_frame,
      excess_m=worst_excess,
      **geometry.describe_pair(worst_pair),
      depth_m=worst_depth,
      affected_samples=int((sample_excess > config.collision_review_m).sum()),
    )
  penetration = max(0.0, -float(lower_array.min()))
  if penetration > config.ground_review_m:
    reason(
      "rejected" if penetration > config.ground_reject_m else "review",
      "ground_penetration",
      depth_m=penetration,
      frame=int(indices[lower_array.argmin()]),
    )
  root_speed = np.linalg.norm(np.gradient(qpos[:, :3], 1 / config.fps, axis=0), axis=1)
  joint_speed = np.sqrt(
    np.square(np.gradient(qpos[:, 7:], 1 / config.fps, axis=0)).mean(1)
  )
  upright = 1 - 2 * (qpos[:, 4] ** 2 + qpos[:, 5] ** 2) > 0.85
  stationary = (
    (lower_array > config.stationary_clearance_m)
    & upright[indices]
    & (root_speed[indices] < 0.15)
    & (joint_speed[indices] < 0.8)
  )
  stationary_spans = spans(stationary)
  for start, end in stationary_spans:
    duration = float(times[end - 1] - times[start])
    possible = min(times[end - 1] + stride / config.fps, len(qpos) / config.fps)
    possible -= max(0.0, times[start] - stride / config.fps)
    if possible >= config.stationary_seconds:
      reason(
        "rejected" if duration >= config.stationary_seconds else "review",
        "stationary_unsupported_pose"
        if duration >= config.stationary_seconds
        else "stationary_duration_uncertain",
        start_frame=int(indices[start]),
        end_frame=int(indices[end - 1]),
        seconds=float(duration),
        possible_seconds=float(possible),
        min_body_clearance_m=float(lower_array[start:end].min()),
      )
  flights = []
  for start, end in spans(lower_array > config.unsupported_clearance_m):
    duration = float(times[end - 1] - times[start])
    possible = min(times[end - 1] + stride / config.fps, len(qpos) / config.fps)
    possible -= max(0.0, times[start] - stride / config.fps)
    acceleration = None
    residual = None
    if end - start >= 4:
      centered = times[start:end] - times[start:end].mean()
      polynomial = np.polyfit(centered, com[start:end], 2)
      acceleration = float(2 * polynomial[0])
      residual = float(
        np.sqrt(np.square(np.polyval(polynomial, centered) - com[start:end]).mean())
      )
    mismatch = acceleration is not None and (
      abs(acceleration - geometry.gravity_z) > config.flight_acceleration_tolerance
    )
    reliable = residual is not None and residual <= config.flight_fit_rmse_m
    flight = {
      "start_frame": int(indices[start]),
      "end_frame": int(indices[end - 1]),
      "seconds": duration,
      "possible_seconds": float(possible),
      "fitted_com_acceleration_m_s2": acceleration,
      "quadratic_com_fit_rmse_m": residual,
      "quadratic_fit_reliable": reliable,
      "has_grounded_before": start > 0,
      "has_grounded_after": end < len(indices),
    }
    flights.append(flight)
    if duration >= config.long_flight_seconds:
      reason(
        "rejected" if mismatch and reliable else "review",
        "long_unsupported_flight",
        **flight,
      )
    elif possible >= config.long_flight_seconds:
      reason("review", "flight_duration_uncertain", **flight)
    elif duration >= 0.4 and mismatch:
      reason("review", "nonballistic_flight", **flight)
    elif duration >= 0.4 and (start == 0 or end == len(indices)):
      reason("review", "flight_truncated_at_clip_boundary", **flight)
  measurements["unsupported_segments"] = flights
  row["scope"] = (
    "full-frame archive/limit/discontinuity checks; sampled static geometry; "
    "label evidence; no dynamic tracking or continuous collision proof"
  )
  return row


def _initialize(xml: str, root: str, config: dict[str, Any], labels: dict[str, Any]):
  global _geometry, _config, _root, _labels
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  _geometry = CollisionGeometry(Path(xml))
  _root, _config, _labels = Path(root), FilterConfig(**config), labels


def _batch(paths: list[str]) -> list[dict[str, Any]]:
  assert _geometry is not None and _config is not None and _root is not None
  rows = []
  for relative in paths:
    try:
      rows.append(screen_motion(_root / relative, _root, _geometry, _config, _labels))
    except Exception as error:
      rows.append(
        {
          "path": relative,
          "sequence_id": sequence_id(relative),
          "status": "review",
          "reasons": [{"code": "screening_error", "error": str(error)}],
        }
      )
  return rows


def atomic_json(path: Path, value: Any) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(temporary, path)


def export_manifests(
  database: sqlite3.Connection,
  output: Path,
  root: Path,
  expected_groups: dict[str, list[str]] | None = None,
) -> None:
  """Publish candidates with complete available-input chunk coverage checks."""
  records = database.execute("SELECT payload FROM results ORDER BY path")
  groups: dict[str, tuple[str, list[str]]] = {}
  handles = {
    status: (output / f"{status}.txt.tmp").open("w")
    for status in ("accepted", "rejected", "review")
  }
  order = {"accepted": 0, "review": 1, "rejected": 2}
  try:
    for (payload,) in records:
      row = json.loads(payload)
      handles[row["status"]].write(str(root / row["path"]) + "\n")
      key = row["sequence_id"]
      previous, paths = groups.get(key, ("accepted", []))
      status = max((previous, row["status"]), key=order.__getitem__)
      groups[key] = status, [*paths, row["path"]]
  finally:
    for handle in handles.values():
      handle.close()
  for status in handles:
    os.replace(output / f"{status}.txt.tmp", output / f"{status}.txt")
  sequence_rows: dict[str, dict[str, Any]] = {}
  for key, (status, paths) in groups.items():
    expected = expected_groups.get(key, []) if expected_groups is not None else []
    matches = set(paths) == set(expected) if expected else False
    numbers = [re.search(r"__clip(\d+)\.npz$", p) for p in expected]
    numbered = [int(n.group(1)) for n in numbers if n is not None]
    contiguous = (
      sorted(numbered) == list(range(len(numbered)))
      if numbered and len(numbered) == len(expected)
      else not numbered
    )
    complete = matches and contiguous
    problems = []
    if expected_groups is None:
      problems.append("available_input_inventory_not_supplied")
    elif not matches:
      problems.append("not_all_available_chunks_screened")
    if not contiguous:
      problems.append("chunk_numbers_do_not_start_at_zero_or_have_gaps")
    if not complete and status == "accepted":
      status = "review"
    sequence_rows[key] = {
      "status": status,
      "paths": paths,
      "available_chunks": len(expected),
      "complete_available_input": complete,
      "coverage_problems": problems,
      "scope": "all available imported chunks; upstream completeness unverified",
    }
  atomic_json(
    output / "sequences.json",
    sequence_rows,
  )
  temporary = output / "accepted_sequences.txt.tmp"
  with temporary.open("w") as stream:
    for row in sequence_rows.values():
      if row["status"] == "accepted" and row["complete_available_input"]:
        for relative in row["paths"]:
          stream.write(str(root / relative) + "\n")
  os.replace(temporary, output / "accepted_sequences.txt")


def run_screen(
  xml: Path,
  root: Path,
  output: Path,
  config: FilterConfig,
  *,
  workers: int = 8,
  max_files: int = 0,
  labels_path: Path | None = None,
  batch_size: int = 16,
) -> dict[str, Any]:
  config.validate()
  if workers < 1 or batch_size < 1 or max_files < 0:
    raise ValueError("workers/batch_size must be positive; max_files nonnegative")
  root, xml = root.resolve(), xml.resolve()
  output = output.resolve()
  if output == root or root in output.parents:
    raise ValueError("Output must be outside the immutable motion library")
  all_files = sorted(root.rglob("*.npz"))
  files = all_files[:max_files] if max_files else all_files
  if not files:
    raise ValueError("No NPZ archives found")
  labels = json.loads(labels_path.read_text()) if labels_path else {}
  inventory: list[tuple[str, int, int]] = [
    (str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
    for p in all_files
  ]
  expected_groups: dict[str, list[str]] = {}
  for relative, _size, _mtime in inventory:
    key = sequence_id(relative)
    expected_groups.setdefault(key, []).append(relative)
  geometry = CollisionGeometry(xml)
  identity = {
    "schema_version": SCHEMA_VERSION,
    "config": asdict(config),
    "model_sha256": geometry.model_sha256,
    "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "inventory_sha256": hashlib.sha256(json.dumps(inventory).encode()).hexdigest(),
    "labels_sha256": hashlib.sha256(
      json.dumps(labels, sort_keys=True).encode()
    ).hexdigest(),
    "total_files": len(files),
    "available_input_files": len(all_files),
    "motion_dir": str(root),
    "xml": str(xml),
  }
  output.mkdir(parents=True, exist_ok=True)
  # Keep the handle alive throughout this run to prevent two resumptions from
  # writing the same SQLite/JSONL files concurrently.
  run_lock = (output / "screening.lock").open("a")
  try:
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError as error:
    run_lock.close()
    raise ValueError("Another screening process owns this output directory") from error
  existing = output / "identity.json"
  if existing.exists() and json.loads(existing.read_text()) != identity:
    raise ValueError("Output identity differs; use a new versioned output directory")
  atomic_json(existing, identity)
  atomic_json(output / "geometry_calibration.json", geometry.calibration())
  database = sqlite3.connect(output / "results.sqlite3")
  database.execute("PRAGMA journal_mode=WAL")
  database.execute("PRAGMA synchronous=FULL")
  database.execute(
    "CREATE TABLE IF NOT EXISTS results (path TEXT PRIMARY KEY, status TEXT, "
    "sequence_id TEXT, payload TEXT NOT NULL)"
  )
  completed = {row[0] for row in database.execute("SELECT path FROM results")}
  remaining = [
    str(p.relative_to(root)) for p in files if str(p.relative_to(root)) not in completed
  ]
  counts = Counter(
    dict(database.execute("SELECT status, COUNT(*) FROM results GROUP BY status"))
  )
  initial_done = len(completed)
  # SQLite is authoritative. Rebuild JSONL to repair a crash between database
  # commit and append, including a partial trailing JSONL record.
  temporary = output / "results.jsonl.tmp"
  with temporary.open("w") as stream:
    for (payload,) in database.execute("SELECT payload FROM results ORDER BY path"):
      stream.write(payload + "\n")
  os.replace(temporary, output / "results.jsonl")
  started, last_export = time.monotonic(), time.monotonic()
  started_utc = time.time()
  total_frames, sampled_frames = 0, 0
  global _stop
  _stop = False

  def request_stop(_signum: int, _frame: Any) -> None:
    global _stop
    _stop = True

  signal.signal(signal.SIGTERM, request_stop)
  signal.signal(signal.SIGINT, request_stop)

  def progress(phase: str) -> dict[str, Any]:
    elapsed = time.monotonic() - started
    done = sum(counts.values())
    rate = (done - initial_done) / max(elapsed, 1e-6)
    result = {
      "phase": phase,
      "pid": os.getpid(),
      "workers": workers,
      "started_unix": started_utc,
      "updated_unix": time.time(),
      "elapsed_seconds": elapsed,
      "completed_files": done,
      "total_files": len(files),
      "counts": dict(counts),
      "clips_per_second": rate,
      "estimated_remaining_seconds": (len(files) - done) / rate if rate else None,
      "new_frames_screened": total_frames,
      "new_geometry_samples": sampled_frames,
      "geometry_samples_per_second": sampled_frames / max(elapsed, 1e-6),
      "library_unchanged": True,
      "training_run_unchanged": True,
      "accepted_scope": "screened candidates; no proof of dynamic feasibility",
    }
    atomic_json(output / "progress.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result

  progress("screening")
  batches = iter(
    [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]
  )
  with (output / "results.jsonl").open("a") as stream:
    with ProcessPoolExecutor(
      max_workers=workers,
      mp_context=multiprocessing.get_context("spawn"),
      initializer=_initialize,
      initargs=(str(xml), str(root), asdict(config), labels),
    ) as pool:
      pending = set()
      for _ in range(workers * 2):
        batch = next(batches, None)
        if batch:
          pending.add(pool.submit(_batch, batch))
      while pending:
        ready, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
        if not ready:
          progress("draining" if _stop else "screening")
        for future in ready:
          rows = future.result()
          for row in rows:
            payload = json.dumps(row, separators=(",", ":"), sort_keys=True)
            database.execute(
              "INSERT INTO results VALUES (?, ?, ?, ?)",
              (row["path"], row["status"], row["sequence_id"], payload),
            )
            stream.write(payload + "\n")
            counts[row["status"]] += 1
            total_frames += row.get("frames", 0)
            sampled_frames += row.get("measurements", {}).get("sampled_frames", 0)
          database.commit()
          stream.flush()
          os.fsync(stream.fileno())
          if not _stop:
            batch = next(batches, None)
            if batch:
              pending.add(pool.submit(_batch, batch))
        if ready:
          progress("draining" if _stop else "screening")
        if time.monotonic() - last_export >= 300:
          export_manifests(database, output, root, expected_groups)
          last_export = time.monotonic()
  export_manifests(database, output, root, expected_groups)
  final = progress("complete" if sum(counts.values()) == len(files) else "interrupted")
  database.close()
  run_lock.close()
  return final


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--xml", type=Path, required=True)
  parser.add_argument("--motion-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument("--sample-hz", type=float, default=10.0)
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--max-files", type=int, default=0)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--labels-json", type=Path)
  parser.add_argument("--allow-unlabeled", action="store_true")
  args = parser.parse_args(argv)
  run_screen(
    args.xml,
    args.motion_dir,
    args.output_dir,
    FilterConfig(
      fps=args.fps, sample_hz=args.sample_hz, allow_unlabeled=args.allow_unlabeled
    ),
    workers=args.workers,
    max_files=args.max_files,
    batch_size=args.batch_size,
    labels_path=args.labels_json,
  )


if __name__ == "__main__":
  main()
