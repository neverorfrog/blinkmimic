"""T2 runtime regression checks. Add --gpu for the 16-world partial-reset check."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from blinkmimic.robot import configure_runtime
from blinkmimic.t2.t2_constants import SPEC


def main():
    args = SimpleNamespace(collect_graph=True, reset_masked=True, eprobe_graph=True, eprobe_async=20, cuda_graph=True)
    assert set(configure_runtime(args, SPEC)) == {'collect_graph','reset_masked','eprobe_graph','eprobe_async'}
    assert args.cuda_graph and not any((args.collect_graph,args.reset_masked,args.eprobe_graph,args.eprobe_async))
    control = SimpleNamespace(collect_graph=True, reset_masked=True, eprobe_graph=True, eprobe_async=20)
    assert configure_runtime(control, SimpleNamespace(has_action_delay=False)) == []
    assert control.collect_graph and control.reset_masked
    from blinkmimic.kinematics import Kinematics
    kin = Kinematics(SPEC, 'cpu')
    assert abs(kin.contact_height - 0.0375) < 1e-8
    print('PASS: delayed robots use safe simulator paths; delay-free paths preserved; T2 contact height',flush=True)
    if '--gpu' not in sys.argv:
        return
    from blinkmimic.dr import XiLayout
    from blinkmimic.env import TrackingSim, make_sim
    from blinkmimic.motion import MotionLib
    from blinkmimic.mdp import MDP
    dev='cuda:0'
    kin=Kinematics(SPEC,dev)
    clip=Path(SPEC.motion_dir)/'run2_subject4_116_124.npz'
    motion=MotionLib([str(clip)],kin,dev,SPEC.load_motion)
    mdp=MDP(kin,motion,torch.tensor(SPEC.default_joint_pos_list(),device=dev),XiLayout(SPEC,dev).dim,SPEC.sensing)
    env=TrackingSim(make_sim(SPEC),16,SPEC,motion,mdp,dev,randomize=False,reset_noise=False)
    ids=torch.arange(16,device=dev)
    env.reset(ids,torch.zeros(16,dtype=torch.long,device=dev))
    for _ in range(12):
        env.step(torch.zeros(16,SPEC.num_joints,device=dev))
    buffers=[a._delay_buffer for a in env.robot.actuators if a._delay_buffer is not None]
    assert buffers, 'No delayed actuators exercised'
    assert all((b._step_count > 0).all() for b in buffers)
    previous=[b._step_count.clone() for b in buffers]
    selected=torch.tensor([0,2,4],device=dev)
    remaining=torch.tensor([1,3,5,6,7,8,9,10,11,12,13,14,15],device=dev)
    env.reset(selected,torch.zeros(3,dtype=torch.long,device=dev))
    for b, before in zip(buffers,previous):
        assert (b._step_count[selected] == 0).all()
        assert torch.equal(b._step_count[remaining], before[remaining])
        assert b._needs_lag[selected].all()
    try:
        env.reset_masked(torch.ones(16,dtype=torch.bool,device=dev))
    except AssertionError as exc:
        assert 'delay' in str(exc)
    else:
        raise AssertionError('Unsafe masked reset accepted delayed actuators')
    mdp.feet=[SPEC.tracked_bodies.index(b) for b in SPEC.foot_bodies]
    r=mdp.reward(env.state(),env.f,env.a_prev,env.a_prev,w_contact=.25)
    assert torch.isfinite(r).all()
    print('PASS: GPU partial reset clears selected delay histories, leaves other worlds untouched, rejects unsafe resets, finite contact reward',flush=True)


if __name__ == '__main__':
    main()
