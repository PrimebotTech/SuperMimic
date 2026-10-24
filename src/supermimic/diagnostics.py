"""Bounded open-loop world-model checks against real Q1 simulator trajectories."""

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from torch import Tensor

from supermimic.data import Q1Motions
from supermimic.models import MotionPolicy, WorldModel, goal_features
from supermimic.robot import Q1Robot, rotation_error
from supermimic.simulator import Q1Simulator


@dataclass
class DiagnosticCfg:
  checkpoint: Path
  output: Path
  device: str = "cpu"
  max_motions: int = 1
  max_steps: int = 160
  num_resets: int = 4
  seed: int = 0
  randomize: bool = False


@dataclass
class EpisodeTrace:
  """One simulator episode, including the last state before any reset."""

  motion: str
  start_frame: int
  states: Tensor
  actions: Tensor
  references: Tensor
  context: Tensor
  terminated: Tensor
  end_reason: str


@dataclass
class PhysicalErrors:
  normalized_physical_rmse: float
  root_position_rmse_m: float
  root_orientation_rmse_rad: float
  joint_angle_rmse_rad: float
  root_linear_velocity_rmse_m_per_s: float
  root_angular_velocity_rmse_rad_per_s: float
  joint_velocity_rmse_rad_per_s: float


@dataclass
class HorizonResult:
  horizon: int
  seconds: float
  windows: int
  status: str
  reason: str | None = None
  errors: PhysicalErrors | None = None
  terminal_labels: int | None = None
  terminal_label_fraction: float | None = None
  termination_bce: float | None = None
  mean_termination_probability: float | None = None
  mean_probability_on_terminal: float | None = None
  mean_probability_on_nonterminal: float | None = None


@dataclass
class EpisodeResult:
  motion: str
  start_frame: int
  evaluated_steps: int
  end_reason: str
  windows: dict[str, int]


@dataclass
class DiagnosticResult:
  robot: str
  checkpoint: str
  checkpoint_iteration: int
  asset_sha256: str
  device: str
  control_dt: float
  max_steps: int
  num_resets: int
  evaluated_steps: int
  randomize: bool
  seed: int
  episodes: list[EpisodeResult]
  horizons: list[HorizonResult]
  notes: list[str]


def window_starts(terminated: Tensor, horizon: int) -> Tensor:
  """Include a terminal endpoint, but never step beyond its transition."""
  if terminated.ndim != 1 or horizon < 1:
    raise ValueError("Expected one-dimensional labels and a positive horizon")
  count = len(terminated) - horizon + 1
  if count <= 0:
    return torch.empty(0, dtype=torch.long, device=terminated.device)
  starts = torch.arange(count, device=terminated.device)
  prefix = torch.cat(
    (torch.zeros(1, dtype=torch.long, device=terminated.device), terminated.long())
  ).cumsum(0)
  # The final transition may terminate. Earlier transitions must survive.
  return starts[prefix[starts + horizon - 1] - prefix[starts] == 0]


@torch.no_grad()
def autoregressive_prediction(
  model: Callable[[Tensor, Tensor, Tensor], Tensor],
  initial_state: Tensor,
  executed_actions: Tensor,
  context: Tensor,
  *,
  advance_context: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
  """Hold the executed action sequence fixed; feed predictions back as states."""
  state = initial_state
  for action in executed_actions.unbind(1):
    state = model(state, action, context)
    if advance_context is not None:
      context = advance_context(context, action)
    if not torch.isfinite(state).all():
      raise FloatingPointError("Nonfinite open-loop world-model prediction")
  return state


def physical_errors(predicted: Tensor, actual: Tensor, scale: Tensor) -> PhysicalErrors:
  """Coordinate RMSEs; quaternion signs align before training-scale RMSE."""
  if predicted.shape != actual.shape or predicted.ndim != 2 or len(predicted) == 0:
    raise ValueError("Expected nonempty matching batches of predicted/actual states")
  if predicted.shape[1] < 57 or len(scale) < 57:
    raise ValueError("Expected the 57-dimensional Q1 physical state and its scale")
  if not torch.isfinite(scale[:57]).all() or not (scale[:57] > 0).all():
    raise ValueError("Physical state scale must be finite and positive")
  orientation = torch.where(
    (predicted[:, 3:7] * actual[:, 3:7]).sum(-1, keepdim=True) < 0,
    -actual[:, 3:7],
    actual[:, 3:7],
  )
  aligned = torch.cat((actual[:, :3], orientation, actual[:, 7:57]), dim=-1)
  error = predicted[:, :57] - aligned

  def rmse(values: Tensor) -> float:
    return float(values.square().mean().sqrt())

  return PhysicalErrors(
    normalized_physical_rmse=rmse(error / scale[:57]),
    root_position_rmse_m=rmse(error[:, :3]),
    root_orientation_rmse_rad=float(
      rotation_error(predicted[:, 3:7], actual[:, 3:7]).mean().sqrt()
    ),
    joint_angle_rmse_rad=rmse(error[:, 7:29]),
    root_linear_velocity_rmse_m_per_s=rmse(error[:, 29:32]),
    root_angular_velocity_rmse_rad_per_s=rmse(error[:, 32:35]),
    joint_velocity_rmse_rad_per_s=rmse(error[:, 35:57]),
  )


@torch.no_grad()
def score_horizon(
  robot: Q1Robot, model: WorldModel, episodes: list[EpisodeTrace], horizon: int
) -> HorizonResult:
  predictions, actuals, labels, logits = [], [], [], []
  for episode in episodes:
    starts = window_starts(episode.terminated, horizon)
    if not len(starts):
      continue
    offsets = starts[:, None] + torch.arange(horizon, device=starts.device)
    context = episode.context[None].expand(len(starts), -1)
    context = model.advance_context(context, episode.states[starts, 57:79])
    predicted = autoregressive_prediction(
      model,
      episode.states[starts],
      episode.actions[offsets],
      context,
      advance_context=model.advance_context,
    )
    end = starts + horizon - 1
    predictions.append(predicted)
    actuals.append(episode.states[end + 1])
    labels.append(episode.terminated[end].float())
    logits.append(model.termination_logits(robot, predicted, episode.references[end]))
  if not predictions:
    return HorizonResult(
      horizon=horizon,
      seconds=0.02 * horizon,
      windows=0,
      status="not_measured",
      reason="No collected episode contains a valid window of this length.",
    )
  prediction, actual = torch.cat(predictions), torch.cat(actuals)
  terminal, logit = torch.cat(labels), torch.cat(logits)
  probability = logit.sigmoid()
  positive = terminal.bool()
  return HorizonResult(
    horizon=horizon,
    seconds=0.02 * horizon,
    windows=len(prediction),
    status="measured",
    errors=physical_errors(prediction, actual, model.scale),
    terminal_labels=int(positive.sum()),
    terminal_label_fraction=float(terminal.mean()),
    termination_bce=float(F.binary_cross_entropy_with_logits(logit, terminal)),
    mean_termination_probability=float(probability.mean()),
    mean_probability_on_terminal=(
      float(probability[positive].mean()) if positive.any() else None
    ),
    mean_probability_on_nonterminal=(
      float(probability[~positive].mean()) if (~positive).any() else None
    ),
  )


@torch.no_grad()
def collect_episodes(
  robot: Q1Robot,
  actor: MotionPolicy,
  motions: Q1Motions,
  simulator: Q1Simulator,
  cfg: DiagnosticCfg,
) -> list[EpisodeTrace]:
  episodes = []
  generator = torch.Generator().manual_seed(cfg.seed)
  ids = torch.tensor([0], device=cfg.device)
  total_steps = 0
  for reset in range(cfg.num_resets):
    clip = reset % len(motions.paths)
    motion = motions.load(clip).to(cfg.device)
    # Start once at each selected clip's beginning, then sample reference resets.
    # Keep at least 32 reference transitions when the clip is long enough.
    last_start = max(0, len(motion) - 33)
    start = (
      0
      if reset < len(motions.paths)
      else int(torch.randint(last_start + 1, (), generator=generator))
    )
    simulator.reset(ids, motion[start : start + 1], randomize=cfg.randomize)
    state = simulator.state()
    observation = robot.observation(state, simulator.context)
    history = observation[:, None].expand(-1, 10, -1).clone()
    states = [state[0].clone()]
    actions, references, terminated = [], [], []
    slot_steps = math.ceil((cfg.max_steps - total_steps) / (cfg.num_resets - reset))
    end_reason = "step_budget"
    for step in range(min(slot_steps, len(motion) - 1 - start)):
      frame = start + step
      future = torch.arange(frame + 1, frame + 11, device=cfg.device).clamp(
        max=len(motion) - 1
      )
      reference = motion[future][None]
      action = actor(history, goal_features(robot, reference))
      next_state = simulator.step(action)
      terminal = robot.terminated(next_state, reference[:, 0])
      states.append(next_state[0].clone())
      actions.append(action[0].clone())
      references.append(reference[0, 0].clone())
      terminated.append(terminal[0].clone())
      total_steps += 1
      if bool(terminal.item()):
        end_reason = "terminated"
        break
      if frame + 1 == len(motion) - 1:
        end_reason = "reference_end"
        break
      observation = robot.observation(next_state, simulator.context)
      history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
    episodes.append(
      EpisodeTrace(
        motion=str(motions.paths[clip].resolve()),
        start_frame=start,
        states=torch.stack(states),
        actions=torch.stack(actions),
        references=torch.stack(references),
        context=simulator.context[0].clone(),
        terminated=torch.stack(terminated),
        end_reason=end_reason,
      )
    )
  return episodes


@torch.no_grad()
def diagnose(cfg: DiagnosticCfg) -> DiagnosticResult:
  if cfg.max_motions not in (1, 2) or cfg.num_resets < 1:
    raise ValueError("Select one or two motions and at least one reset")
  if cfg.max_steps < cfg.num_resets:
    raise ValueError("The total step budget must allow one step per reset")
  if cfg.device != "cpu" and not cfg.device.startswith("cuda"):
    raise ValueError("Use CPU MuJoCo or CUDA MuJoCo Warp")
  if cfg.output.exists():
    raise FileExistsError(f"Diagnostic output already exists: {cfg.output}")
  torch.set_num_threads(min(torch.get_num_threads(), 4))
  torch.manual_seed(cfg.seed)
  checkpoint = torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=True)
  config, identity = checkpoint["config"], checkpoint["identity"]
  robot = Q1Robot(
    Path(config["robot_xml"]),
    modeled_action_delay=config.get("modeled_action_delay", False),
  ).to(cfg.device)
  if (
    robot.sha256 != identity["xml_sha256"]
    or robot.asset_sha256 != identity["asset_sha256"]
    or robot.joint_names != tuple(identity["joint_names"])
  ):
    raise ValueError("Q1 XML/mesh/joint identity differs from the checkpoint")
  if (
    robot.physical_dim != 57
    or robot.state_dim != identity["state_dim"]
    or robot.obs_dim != identity["obs_dim"]
    or robot.context_dim != identity["context_dim"]
    or identity["control_dt"] != 0.02
  ):
    raise ValueError("Checkpoint does not describe the 50 Hz Q1 state interface")
  robot.load_state_dict(checkpoint["robot"])
  model = WorldModel(
    robot,
    tuple(identity["model_hidden"]),
    activation=config.get("network_activation", "elu"),
    representation=config.get("dynamics_representation", "legacy"),
  ).to(cfg.device)
  actor = MotionPolicy(
    robot,
    tuple(identity["actor_hidden"]),
    activation=config.get("network_activation", "elu"),
  ).to(cfg.device)
  model.load_state_dict(checkpoint["model"])
  actor.load_state_dict(checkpoint["actor"])
  model.eval()
  actor.eval()
  paths = [Path(p) for p in identity["motion_paths"]][: cfg.max_motions]
  motions = Q1Motions(robot, paths, dt=identity["control_dt"], device=cfg.device)
  simulator = Q1Simulator(
    robot,
    1,
    cfg.device,
    config["action_scale"],
    use_tn=config.get("use_tn", False),
    physics_timestep=config.get("physics_timestep"),
  )
  episodes = collect_episodes(robot, actor, motions, simulator, cfg)
  horizons = [score_horizon(robot, model, episodes, h) for h in (1, 8, 32)]
  result = DiagnosticResult(
    robot="q1_v2",
    checkpoint=str(cfg.checkpoint.resolve()),
    checkpoint_iteration=int(checkpoint["iteration"]),
    asset_sha256=robot.asset_sha256,
    device=cfg.device,
    control_dt=motions.dt,
    max_steps=cfg.max_steps,
    num_resets=cfg.num_resets,
    evaluated_steps=sum(len(e.actions) for e in episodes),
    randomize=cfg.randomize,
    seed=cfg.seed,
    episodes=[
      EpisodeResult(
        motion=e.motion,
        start_frame=e.start_frame,
        evaluated_steps=len(e.actions),
        end_reason=e.end_reason,
        windows={str(h): len(window_starts(e.terminated, h)) for h in (1, 8, 32)},
      )
      for e in episodes
    ],
    horizons=horizons,
    notes=[
      "Actions come from the current deterministic actor in real Q1 simulation; "
      "open-loop prediction reuses those actions without teacher forcing.",
      "Metrics compare endpoints at each horizon, using overlapping windows "
      "within episodes. Terminal endpoints are included; resets are never crossed.",
      "Physical RMSE uses the training model's first 57 scale values and aligns "
      "quaternion hemispheres. Other RMSEs average coordinates except SO(3) angle.",
      "Termination labels indicate the paper's height/tilt failure rule. Reference "
      "end and step budget are boundaries, not positive termination labels.",
      "Observation and action noise are disabled. Reference states only initialize "
      "resets and provide actor goals/termination targets, never trajectory labels.",
      "This bounded diagnostic is not a convergence or motion-reproduction result.",
    ],
  )
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  with cfg.output.open("x") as stream:
    stream.write(json.dumps(asdict(result), indent=2, allow_nan=False) + "\n")
  print(json.dumps(asdict(result), allow_nan=False), flush=True)
  return result


def main() -> None:
  diagnose(tyro.cli(DiagnosticCfg))


if __name__ == "__main__":
  main()
