"""CPU checks for the Colosseum T1/T2 adapters; run with Colosseum's Pixi environment.

    python tools/test_t2.py [t1|t2]
"""

import importlib
import re
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch

from blinkmimic.kinematics import Kinematics
from blinkmimic.mirror import Mirror
from blinkmimic.motion import MotionLib
from blinkmimic.rotations import quat_to_mat
from blinkmimic.robot import get_robot

ROBOT = sys.argv[1] if len(sys.argv) > 1 else "t2"
SPEC = get_robot(ROBOT)
load_npz = SPEC.load_motion
_constants = importlib.import_module(f"colosseum.robots.{ROBOT}.constants")
FLIP_SIGN_JOINT_NAMES, SYMMETRY_JOINT_NAMES = _constants.FLIP_SIGN_JOINT_NAMES, _constants.SYMMETRY_JOINT_NAMES


def body_partner(name: str) -> str:
    """left_/right_, Left/Right and the T1 arm links AL*/AR* swapped."""
    swap = {"left_": "right_", "right_": "left_", "Left": "Right", "Right": "Left", "AL": "AR", "AR": "AL"}
    return re.sub(r"left_|right_|Left|Right|^AL|^AR", lambda m: swap[m.group(0)], name)


def main():
    torch.manual_seed(0)
    model = mujoco.MjModel.from_xml_path(SPEC.mjcf_path)
    assert [model.joint(j).name for j in range(1, model.njnt)] == list(SPEC.joint_names)
    kin = Kinematics(SPEC, "cpu")
    mirror = Mirror(SPEC, {}, "cpu")
    assert mirror.perm.tolist() == [SPEC.joint_names.index(SYMMETRY_JOINT_NAMES[n]) for n in SPEC.joint_names]
    assert mirror.sign.tolist() == [-1 if n in FLIP_SIGN_JOINT_NAMES else 1 for n in SPEC.joint_names]
    B = 16
    p = torch.randn(B, 3)
    quat = torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)
    R = quat_to_mat(quat)
    v, w = torch.randn(B, 3), torch.randn(B, 3)
    q = kin.q_lo + torch.rand(B, SPEC.num_joints) * (kin.q_hi - kin.q_lo)
    qd = torch.randn_like(q)
    S = (p, R, v, w, q, qd)
    bp, bR, bv, bw = kin.forward(*S)
    errors = np.zeros(4)
    data = mujoco.MjData(model)
    for i in range(B):
        data.qpos[:] = np.concatenate((p[i], quat[i], q[i]))
        data.qvel[:] = np.concatenate((v[i], R[i].T @ w[i], qd[i]))
        mujoco.mj_forward(model, data)
        for j, name in enumerate(SPEC.tracked_bodies):
            b = model.body(name).id
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)
            for k, (a, expected) in enumerate(zip((bp[i,j], bR[i,j], bv[i,j], bw[i,j]),
                    (data.xpos[b], data.xmat[b].reshape(3,3), vel[3:], vel[:3]))):
                errors[k] = max(errors[k], np.abs(a.numpy() - expected).max())
    assert max(errors) < 1e-4, errors
    mirrored = kin.forward(*mirror.state(S))
    body_perm = [SPEC.tracked_bodies.index(body_partner(n)) for n in SPEC.tracked_bodies]
    assert torch.allclose(mirrored[0], bp[:, body_perm] @ mirror.M, atol=1e-5)
    assert torch.allclose(mirrored[1], mirror.M @ bR[:, body_perm] @ mirror.M, atol=1e-5)
    clip = next(Path(SPEC.motion_dir).glob("*.npz"))
    lib = MotionLib([str(clip)], kin, "cpu", load_npz)
    assert torch.isfinite(lib.body_p).all()
    # A loader must reject silently permuted joint arrays.
    with np.load(clip) as src, tempfile.TemporaryDirectory() as tmp:
        payload = dict(src)
        payload["joint_names"] = payload["joint_names"][::-1]
        bad = Path(tmp) / "bad.npz"
        np.savez(bad, **payload)
        try:
            load_npz(str(bad))
        except ValueError:
            pass
        else:
            raise AssertionError("permuted joints accepted")
    q = q.requires_grad_()
    kin.forward(p, R, v, w, q, qd)[0].square().sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().max() > 0
    print("PASS: MuJoCo FK pose/velocity errors", errors)
    print("PASS: mirror geometry, FK gradients, NPZ loader and motion library", lib.num_frames, "frames")


if __name__ == "__main__":
    main()
