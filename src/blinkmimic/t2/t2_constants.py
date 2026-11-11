"""T2 expert-data tracking in Colosseum's robot and actuator conventions."""

import numpy as np

from colosseum.robots.actuator import action_scale
from colosseum.robots.t2.actuators import ACTUATOR_PARAMETERS
from colosseum.robots.t2.constants import (
    BASE_BODY_NAME, FOOT_BODY_NAMES, HOME_QPOS, JOINT_NAMES, XML, get_robot_cfg as _robot_cfg,
)
from colosseum.utils import project_root

from blinkmimic.dr import DomainRandCfg
from blinkmimic.robot import MotionArrays, RobotSpec
from blinkmimic.sensing import SensingCfg

TRACKED_BODIES = (
    "trunk", "waist_yaw_link", "head_yaw_link", "head_pitch_link",
    "left_shoulder_pitch_link", "left_elbow_pitch_link", "left_wrist_roll_link",
    "right_shoulder_pitch_link", "right_elbow_pitch_link", "right_wrist_roll_link",
    "left_hip_roll_link", "left_knee_pitch_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_pitch_link", "right_ankle_roll_link",
)


def get_robot_cfg():
    return _robot_cfg(self_collision=True)


def load_npz(path: str) -> MotionArrays:
    """Read process_motions NPZs; root poses are link poses, quaternions wxyz."""
    with np.load(path, allow_pickle=False) as data:
        if list(data["joint_names"]) != JOINT_NAMES:
            raise ValueError(f"{path}: joint order does not match T2")
        root = list(data["body_names"]).index(BASE_BODY_NAME)
        p = data["body_pos_w"][:, root].copy()
        quat = data["body_quat_w"][:, root].copy()
        q = data["joint_pos"].copy()
        fps = float(data["fps"].reshape(-1)[0])
    if len(q) < 2 or p.shape != (len(q), 3) or quat.shape != (len(q), 4) or q.shape != (len(q), len(JOINT_NAMES)):
        raise ValueError(f"{path}: invalid T2 motion shapes")
    if not np.isfinite(fps) or fps <= 0 or not all(np.isfinite(a).all() for a in (p, quat, q)):
        raise ValueError(f"{path}: nonfinite motion or invalid fps")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-4):
        raise ValueError(f"{path}: root quaternions must be normalized")
    return p, quat, q, fps


SPEC = RobotSpec(
    name="t2",
    mjcf_path=str(XML),
    joint_names=tuple(JOINT_NAMES),
    tracked_bodies=TRACKED_BODIES,
    foot_bodies=FOOT_BODY_NAMES,
    contact_height=0.0375,  # neutral sole bottom is 32.5 mm below the ankle, plus 5 mm clearance
    default_joint_pos=HOME_QPOS,
    action_scale=action_scale(ACTUATOR_PARAMETERS),
    com_body=BASE_BODY_NAME,
    com_range=((0.0, 0.0),) * 3,
    robot_cfg=get_robot_cfg,
    motion_dir=str(project_root() / "motions/t2/locomotion_t2_amp/clean"),
    motion_glob="*.npz",
    load_motion=load_npz,
    dr=DomainRandCfg(friction_range=(1.0, 1.0), joint_offset={".*": 0.0}),
    sensing=SensingCfg(noise_joint_pos=0, noise_joint_vel=0, noise_ang_vel=0,
                       noise_gravity=0, noise_ref_ori=0),
    njmax=1500,
    nconmax=45,
    ccd_iterations=500,
)
