"""T1 expert-data tracking in Colosseum's robot and actuator conventions."""

import numpy as np

from colosseum.robots.actuator import action_scale
from colosseum.robots.t1.actuators import ACTUATOR_PARAMETERS
from colosseum.robots.t1.constants import (
    BASE_BODY_NAME, FLIP_SIGN_JOINT_NAMES, FOOT_BODY_NAMES, HOME_QPOS, JOINT_NAMES,
    SYMMETRY_JOINT_NAMES, XML, get_robot_cfg as _robot_cfg,
)
from colosseum.utils import project_root

from blinkmimic.dr import DomainRandCfg
from blinkmimic.robot import MotionArrays, RobotSpec
from blinkmimic.sensing import SensingCfg

# The T2 adapter's bodies on the T1: AL1/AL3/left_hand_link are the shoulder pitch,
# elbow pitch and hand links.
TRACKED_BODIES = (
    "Trunk", "Waist", "H1", "H2",
    "AL1", "AL3", "left_hand_link",
    "AR1", "AR3", "right_hand_link",
    "Hip_Roll_Left", "Shank_Left", "left_foot_link",
    "Hip_Roll_Right", "Shank_Right", "right_foot_link",
)


def get_robot_cfg():
    # As t1-locomotion-amp.
    return _robot_cfg(self_collision=True, arm_torso_collision=True)


def load_npz(path: str) -> MotionArrays:
    """Read process_motions NPZs; root poses are link poses, quaternions wxyz."""
    with np.load(path, allow_pickle=False) as data:
        if list(data["joint_names"]) != JOINT_NAMES:
            raise ValueError(f"{path}: joint order does not match T1")
        root = list(data["body_names"]).index(BASE_BODY_NAME)
        p = data["body_pos_w"][:, root].copy()
        quat = data["body_quat_w"][:, root].copy()
        q = data["joint_pos"].copy()
        fps = float(data["fps"].reshape(-1)[0])
    if len(q) < 2 or p.shape != (len(q), 3) or quat.shape != (len(q), 4) or q.shape != (len(q), len(JOINT_NAMES)):
        raise ValueError(f"{path}: invalid T1 motion shapes")
    if not np.isfinite(fps) or fps <= 0 or not all(np.isfinite(a).all() for a in (p, quat, q)):
        raise ValueError(f"{path}: nonfinite motion or invalid fps")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-4):
        raise ValueError(f"{path}: root quaternions must be normalized")
    return p, quat, q, fps


SPEC = RobotSpec(
    name="t1",
    mjcf_path=str(XML),
    joint_names=tuple(JOINT_NAMES),
    tracked_bodies=TRACKED_BODIES,
    foot_bodies=FOOT_BODY_NAMES,
    contact_height=0.035,  # foot box sole is 30 mm below the foot link, plus 5 mm clearance
    default_joint_pos=HOME_QPOS,
    action_scale=action_scale(ACTUATOR_PARAMETERS),
    com_body=BASE_BODY_NAME,
    com_range=((0.0, 0.0),) * 3,
    robot_cfg=get_robot_cfg,
    motion_dir=str(project_root() / "motions/t1/locomotion_t1_amp/clean"),
    motion_glob="*.npz",
    load_motion=load_npz,
    dr=DomainRandCfg(friction_range=(1.0, 1.0), joint_offset={".*": 0.0}),
    sensing=SensingCfg(noise_joint_pos=0, noise_joint_vel=0, noise_ang_vel=0,
                       noise_gravity=0, noise_ref_ori=0),
    # T1 names are Left_*/Right_* and its roll/yaw joints are not all *_roll_*/*_yaw_*.
    mirror_joints=SYMMETRY_JOINT_NAMES,
    flip_sign_joints=tuple(FLIP_SIGN_JOINT_NAMES),
    njmax=1500,
    nconmax=96,  # as t1-locomotion-amp
    ccd_iterations=500,
)
