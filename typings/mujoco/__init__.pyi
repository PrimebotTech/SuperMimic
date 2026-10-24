"""Minimal typed interface to the native MuJoCo APIs used by this project."""

from typing import Any

import numpy as np

Array = np.ndarray[Any, np.dtype[Any]]

class _NamedBody:
  id: int
  name: str

class _NamedJoint:
  name: str

class _Options:
  timestep: float
  tolerance: float

class MjModel:
  nq: int
  nv: int
  nu: int
  na: int
  neq: int
  nmocap: int
  nbody: int
  opt: _Options
  qpos0: Array
  jnt_type: Array
  jnt_qposadr: Array
  jnt_range: Array
  jnt_pos: Array
  jnt_axis: Array
  actuator_trnid: Array
  actuator_trntype: Array
  actuator_gaintype: Array
  actuator_biastype: Array
  actuator_gear: Array
  actuator_ctrllimited: Array
  actuator_ctrlrange: Array
  body_parentid: Array
  body_jntadr: Array
  body_jntnum: Array
  body_pos: Array
  body_quat: Array
  body_ipos: Array
  body_mass: Array
  body_subtreemass: Array
  body_gravcomp: Array
  geom_friction: Array
  geom_solref: Array
  geom_bodyid: Array
  geom_type: Array
  mesh_vert: Array
  mesh_face: Array
  @staticmethod
  def from_xml_path(path: str) -> MjModel: ...
  def body(self, key: str | int) -> _NamedBody: ...
  def joint(self, key: str | int) -> _NamedJoint: ...

class MjData:
  qpos: Array
  qvel: Array
  ctrl: Array
  xpos: Array
  xquat: Array
  def __init__(self, model: MjModel) -> None: ...

class mjtGeom:
  mjGEOM_SPHERE: int
  mjGEOM_PLANE: int

class MjvGeom: ...

class MjvScene:
  ngeom: int
  maxgeom: int
  geoms: Any

class mjtJoint:
  mjJNT_FREE: int
  mjJNT_HINGE: int

class mjtTrn:
  mjTRN_JOINT: int

class mjtGain:
  mjGAIN_FIXED: int

class mjtBias:
  mjBIAS_NONE: int

def mj_differentiatePos(
  model: MjModel, velocity: Array, dt: float, first: Array, second: Array
) -> None: ...
def mj_integratePos(
  model: MjModel, position: Array, velocity: Array, dt: float
) -> None: ...
def mj_resetData(model: MjModel, data: MjData) -> None: ...
def mj_setConst(model: MjModel, data: MjData) -> None: ...
def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_step(model: MjModel, data: MjData) -> None: ...
def mjv_initGeom(
  geom: MjvGeom, type: int, size: Array, pos: Array, mat: Array, rgba: Array
) -> None: ...
