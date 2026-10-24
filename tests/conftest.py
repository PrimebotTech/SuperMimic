"""Shared Q1 facts for the synthetic fixtures.

The PD table and torso mass event resolve joint and body names from the XML.
These are the Q1 V2 names in XML order.
"""

Q1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
)


def link_name(joint: str) -> str:
  """The body a joint drives, matching the XML's own pairing."""
  # waist_yaw_joint drives the torso in the Q1 V2 XML, which Q1Robot looks up
  # by name for the tilt termination.
  return "torso_link" if joint == "waist_yaw_joint" else joint[: -len("joint")] + "link"
