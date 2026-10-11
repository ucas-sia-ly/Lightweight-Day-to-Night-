"""Prepare audited RobotCar-only inputs and aggregate the ablation results."""
import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
from a2b_lightvpr.config import load_config
from a2b_lightvpr.checkpointing import file_sha256, write_json


def prepare():
    c = load_config('configs/experiments/single_dataset/robotcar_only_cdnd.yaml')
    manifest = list(csv.DictReader(open(c.data.robotcar_manifest)))
    paths = {s: set() for s in ('train','val','test')}
    traverses = {s: set() for s in paths}
    train_domains = {'day': set(), 'night': set()}
    for r in manifest:
        source_path = Path(r['source_path'])
        if not source_path.is_absolute():
            base = next(p.parent for p in Path(c.data.robotcar_manifest).resolve().parents if p.name == 'dataset')
            source_path = base / source_path
        resolved = str(source_path.resolve())
        paths[r['split']].add(resolved)
        traverses[r['split']].add(r['traverse'])
        if r['split']=='train': train_domains[r['domain']].add(resolved)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if paths[a]&paths[b] or traverses[a]&traverses[b]:
            raise ValueError(f'RobotCar split leakage: {a}/{b}')
    prepared={}
    for name in ('generator_train','vpr_train'):
        source=Path('data/manifests_clean_night')/f'{name}.csv'
        with source.open() as f:
            reader=csv.DictReader(f);fields=reader.fieldnames
            rows=[r for r in reader if r['dataset']=='robotcar' and r['split']=='train' and r.get('valid','1') not in ('0','false')]
        if not rows: raise ValueError(f'Empty {name}')
        for r in rows:
            for field,domain in [('source_path','day'),('target_path','night')]:
                if r.get(field):
                    p=Path(r[field])
                    if not p.is_file(): raise FileNotFoundError(p)
                    if str(p.resolve()) not in train_domains[domain]:
                        raise ValueError(f'Not a RobotCar training {domain} image: {p}')
        prepared[name]=(fields,rows,source)
    counts=Counter(r['place_id'] for r in prepared['vpr_train'][1])
    if min(counts.values())<2: raise ValueError('VPR place has fewer than two views')
    # Validate all inputs before committing either manifest.
    for name,(fields,rows,source) in prepared.items():
        buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();w.writerows(rows)
        content=buf.getvalue();dest=Path('data/manifests_robotcar_only')/f'{name}.csv'
        if dest.exists() and dest.read_bytes()!=content.encode(): raise ValueError(f'Existing manifest changed: {dest}')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): dest.write_bytes(content.encode())
    gen=prepared['generator_train'][1]
    audit=dict(generator_day=len({r['source_path'] for r in gen if r['source_path']}),
        generator_night=len({r['target_path'] for r in gen if r['target_path']}),
        vpr_images=sum(counts.values()),vpr_places=len(counts),place_size_histogram=dict(Counter(counts.values())),
        traverses={s:sorted(t) for s,t in traverses.items()},resolved_path_overlap=False,
        hashes={str(p):file_sha256(p) for p in [Path(c.data.robotcar_manifest),Path(c.data.robotcar_protocol),
            *[v[2] for v in prepared.values()],Path(c.data.generator_manifest),Path(c.data.vpr_source_manifest)]})
    write_json(audit,'data/manifests_robotcar_only/audit.json');print(json.dumps(audit,indent=2))


def summarize():
    root=Path('runs/vpr/robotcar_only_512');models={}
    for name in ('official','day_only','day_fake','cdnd'):
        p=root/'official'/ 'summary.json' if name=='official' else root/f'{name}_seed0/final_suite/summary.json'
        d=json.loads(p.read_text());models[name]={'source':str(p),'identity':{k:v for k,v in d.items() if k!='datasets'},'datasets':{}}
        cfg=load_config('configs/experiments/single_dataset/robotcar_only_day_only.yaml' if name=='official' else f'configs/experiments/single_dataset/robotcar_only_{name}.yaml')
        ckpt=Path(cfg.vpr.init_checkpoint) if name=='official' else root/f'{name}_seed0/model_best.pt'
        models[name]['checkpoint']=str(ckpt.resolve())
        models[name]['checkpoint_sha256']=file_sha256(ckpt)
        models[name]['input_size']=[320,320]
        models[name]['initialization']=cfg.vpr.init_checkpoint
        for protocol in ('robotcar_project','robotcar_day'):
            m=d['datasets'][protocol]
            models[name]['datasets'][protocol]={k:m[k] for k in ('recalls','valid_queries','total_queries','query_protocol_fingerprint')}
    for protocol in ('robotcar_project','robotcar_day'):
        signatures={(m['datasets'][protocol]['query_protocol_fingerprint'],m['datasets'][protocol]['valid_queries']) for m in models.values()}
        if len(signatures)!=1: raise ValueError(f'Inconsistent evaluation protocol: {protocol}')
    delta={}
    for label,a,b in [('fake_minus_day','day_only','day_fake'),('cdnd_minus_fake','day_fake','cdnd')]:
        delta[label]={p:{k:100*(models[b]['datasets'][p]['recalls'][k]-models[a]['datasets'][p]['recalls'][k]) for k in ('1','5','10')} for p in ('robotcar_project','robotcar_day')}
    write_json({'models':models,'delta_percentage_points':delta,'seed':0},root/'ablation_summary.json')
    with (root/'ablation_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Method','Protocol','R@1 %','R@5 %','R@10 %','Valid queries'])
        for n,m in models.items():
            for p,v in m['datasets'].items(): w.writerow([n,p,*[100*v['recalls'][k] for k in ('1','5','10')],v['valid_queries']])
    print(root/'ablation_summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','summarize']);args=p.parse_args()
    prepare() if args.stage=='prepare' else summarize()
