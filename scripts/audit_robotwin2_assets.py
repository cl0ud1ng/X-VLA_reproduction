#!/usr/bin/env python3
"""Audit the downloaded RoboTwin embodiment bundle against the experiment contract."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import yaml

DOMAINS = {
    "aloha-agilex": {"domain_id": 0, "asset": "aloha-agilex", "dual": True},
    "arx-x5": {"domain_id": 1, "asset": "ARX-X5", "dual": False},
    "piper": {"domain_id": 2, "asset": "piper", "dual": False},
}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--asset-root',type=Path,default=Path('assets/robotwin/embodiments'))
    ap.add_argument('--zip',type=Path,default=Path('assets/embodiments.zip'))
    ap.add_argument('--output',type=Path,default=Path('outputs/robotwin_ft/assets_preflight.json'))
    args=ap.parse_args()
    results={}
    for name, spec in DOMAINS.items():
        root=args.asset_root/spec['asset']
        cfg_path=root/'config.yml'
        if not cfg_path.exists(): raise FileNotFoundError(cfg_path)
        cfg=yaml.safe_load(cfg_path.read_text())
        arm=cfg.get('arm_joints_name')
        grip=cfg.get('gripper_name')
        if not isinstance(arm,list) or len(arm)!=2 or any(not isinstance(x,list) or len(x)!=6 for x in arm):
            raise ValueError(f'{name}: expected two 6-DoF arm joint lists')
        if not isinstance(grip,list) or len(grip)!=2:
            raise ValueError(f'{name}: expected two gripper profiles')
        urdf=root/cfg['urdf_path'].lstrip('./')
        if not urdf.exists(): raise FileNotFoundError(urdf)
        urdf_text=urdf.read_text(errors='ignore')
        mesh_refs=sorted(set(re.findall(r'(?:filename|mesh)=["\']([^"\']+)',urdf_text)))
        missing=[]
        for ref in mesh_refs:
            if ref.startswith('$('):
                continue
            if ref.startswith('package://'):
                rel=Path(ref.split('/',3)[-1])
            else: rel=Path(ref)
            if not (urdf.parent/rel).exists() and not (root/rel).exists() and not (root/rel.name).exists(): missing.append(ref)
        if missing: raise FileNotFoundError(f'{name}: missing URDF mesh refs {missing[:8]}')
        pose=cfg.get('robot_pose')
        if not isinstance(pose,list) or not pose or len(pose[0])!=7: raise ValueError(f'{name}: invalid robot_pose')
        results[name]={
            'domain_id':spec['domain_id'],'asset_dir':str(root.resolve()),
            'config':str(cfg_path.resolve()),'urdf':str(urdf.resolve()),
            'arm_dim':[len(x) for x in arm],'gripper_dim':[1,1],
            'dual_arm_asset_flag':bool(cfg.get('dual_arm')),
            'task_config_semantics':'[embodiment]' if spec['dual'] else f'[{spec["asset"]}, {spec["asset"]}, distance]',
            'robot_pose':pose[0],'mesh_refs':len(mesh_refs),'missing_mesh_refs':missing,
            'piper_or_arx_gripper_profiles':grip,
        }
    if args.zip.exists():
        zip_hash=sha256(args.zip); zip_size=args.zip.stat().st_size
    else: zip_hash=None; zip_size=None
    out={'asset_source':'TianxingChen/RoboTwin2.0@main/embodiments.zip','asset_zip':str(args.zip.resolve()),'asset_zip_size':zip_size,'asset_zip_sha256':zip_hash,'domains':results,'contract':{'arm_dim':[6,6],'gripper_dim':[1,1],'piper_must_be_dual':True}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
