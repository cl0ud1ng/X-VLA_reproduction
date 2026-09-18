#!/usr/bin/env python3
"""Full audit for the normalized RoboTwin2 fine-tuning data."""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import h5py, numpy as np
from scipy.spatial.transform import Rotation
try:
    from robotwin2_decode import decode_image_bit
except Exception as exc:
    raise RuntimeError("scripts/robotwin2_decode.py is required") from exc


DOMAINS={'aloha-agilex':0,'arx-x5':1,'piper':2}
TASKS=['beat_block_hammer','stack_blocks_two','move_can_pot','open_microwave','place_dual_shoes']

def rot_sf(q): return Rotation.from_quat(np.asarray(q)[..., [1,2,3,0]])
def pose_matrix(p):
 T=np.eye(4); T[:3,:3]=rot_sf(p[3:]).as_matrix(); T[:3,3]=p[:3]; return T
def roundtrip_pose(p, base):
 T=pose_matrix(p); B=pose_matrix(base); base_pose=np.linalg.inv(B)@T; back=B@base_pose
 return float(np.max(np.abs(T[:3,3]-back[:3,3]))), float(np.max(np.abs(T[:3,:3]-back[:3,:3])))

def sixd_matrix(v):
 a1=np.asarray(v)[0:6:2]; a2=np.asarray(v)[1:6:2]
 b1=a1/np.linalg.norm(a1); b2=a2-np.dot(b1,a2)*b1; b2=b2/np.linalg.norm(b2); b3=np.cross(b1,b2)
 return np.stack([b1,b2,b3],axis=1)

def assert_scalar_first_6d_roundtrip():
 quats=np.array([[1.,0.,0.,0.],[0.70710678,0.,0.70710678,0.],[0.5,0.5,0.5,0.5]])
 for q in quats:
  q=q/np.linalg.norm(q); matrix=rot_sf(q).as_matrix(); v=matrix[:,:2].reshape(6); err=np.max(np.abs(matrix-sixd_matrix(v)))
  assert err<1e-5, (q,err)

def audit_h5(path:Path, pair:dict, checked_image_shapes:set):
 with h5py.File(path,'r') as f:
  version=f['data_format_version'][()]; version=version.decode() if isinstance(version,bytes) else str(version)
  assert version=='v1.0', (path,version)
  freq=float(np.asarray(f['additional_info/frequency']))
  assert math.isfinite(freq) and freq>0
  n=len(f['state/left_ee_poses']); assert n==len(f['action/left_ee_poses']) and n>=1
  for side in ('left','right'):
   for phase in ('state','action'):
    assert f[f'{phase}/{side}_ee_poses'].shape==(n,7)
    assert f[f'{phase}/{side}_ee_joint_states'].shape==(n,1)
    assert f[f'{phase}/{side}_arm_joint_states'].shape==(n,6)
    assert np.isfinite(f[f'{phase}/{side}_ee_poses'][:]).all()
  for _,cam in [('head','cam_head'),('left','cam_left_wrist'),('right','cam_right_wrist')]:
   ds=f[f'vision/{cam}/colors']; assert len(ds)==n
   shape=tuple(int(x) for x in np.asarray(f[f'vision/{cam}/shape']))
   img=np.asarray(decode_image_bit(ds[0])); assert img.ndim==3 and img.shape[-1]==3 and img.dtype==np.uint8
   assert tuple(img.shape)==shape
   checked_image_shapes.add(shape)
  # State/action must be the official adjacent-frame representation.
  for side in ('left','right'):
   state=np.asarray(f[f'state/{side}_ee_poses']); action=np.asarray(f[f'action/{side}_ee_poses'])
   if n>1:
    assert np.max(np.abs(state[1:,:3]-action[:-1,:3]))<1e-5
    r0=rot_sf(state[1:,3:]); r1=rot_sf(action[:-1,3:]); assert np.max(np.abs((r0.inv()*r1).as_rotvec()))<1e-5
  for side,base in [('left',pair['base_pose_left']),('right',pair['base_pose_right'])]:
   for pose in np.asarray(f[f'state/{side}_ee_poses'])[:3]:
    ep,er=roundtrip_pose(pose,base); assert ep<1e-5 and er<1e-5
 return n,freq

def draw(prob, keys, rng):
 vals=np.array([prob[k] for k in keys]); vals=vals/vals.sum(); sample=rng.choice(len(keys),size=10000,p=vals); count=np.bincount(sample,minlength=len(keys))/10000
 return float(np.max(np.abs(count-vals)))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,default=Path('outputs/robotwin_ft/manifests/official_clean_50/total.json')); ap.add_argument('--max-files',type=int,default=0); args=ap.parse_args()
 total=json.loads(args.manifest.read_text()); pairs=total['pairs']; assert len(pairs)==15
 assert set(total['tasks'])==set(TASKS); assert {v['domain_id'] for v in total['domains'].values()}=={0,1,2}
 assert_scalar_first_6d_roundtrip()
 checked=set(); total_h5=0; total_windows=0
 for pair in pairs:
  assert pair['episodes']==50 and pair['split']=='train' and pair['num_actions']==30 and pair['qdur_sec']==1.0
  domain=next(k for k,v in total['domains'].items() if v['domain_id']==pair['domain_id'])
  assert pair['task'] in TASKS
  win_path=args.manifest.parent/f'{domain}__{pair["task"]}.windows.jsonl'
  windows=[json.loads(x) for x in win_path.read_text().splitlines() if x.strip()]
  assert len(windows)==pair['num_action_observation_windows']
  assert sum(1 for x in windows if x['terminal_hold'])==pair['num_terminal_hold_windows']
  for x in windows:
   assert x['domain_id']==pair['domain_id'] and x['task']==pair['task'] and x['num_actions']==30 and x['qdur_sec']==1.0
   assert math.isfinite(x['timestamp_sec']) and x['frame_idx']>=0
  for idx,p in enumerate(pair['datalist']):
   if args.max_files and total_h5>=args.max_files: break
   n,freq=audit_h5(Path(p),pair,checked); assert freq==pair['frequency_hz']; total_h5+=1
  total_windows+=len(windows)
 assert total_h5==(750 if not args.max_files else args.max_files)
 assert total_windows==sum(total['N_dt'].values())
 assert total_windows==149265
 assert sum(total['N_d'].values())==total_windows
 keys=sorted(total['N_dt'])
 for formula in ('domain_balanced','tempered_T2'):
  prob=total['sampling'][formula]['probabilities']; assert abs(sum(prob.values())-1)<1e-9
  err=draw(prob,keys,np.random.default_rng(0)); print(formula,'max_abs_error_10000=',err); assert err<0.02
 print('pairs=15 hdf5=',total_h5,'windows=',total_windows,'image_shapes=',sorted(checked))
 print('N_dt=',json.dumps(total['N_dt'],sort_keys=True)); print('N_d=',json.dumps(total['N_d'],sort_keys=True))
 print('PASS')
if __name__=='__main__': main()
