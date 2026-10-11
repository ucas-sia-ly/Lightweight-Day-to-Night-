"""Reuse validated per-dataset generated images without retraining QS-Attn."""
import argparse,csv,json
from pathlib import Path
from prepare_multidataset_qsattn import write_csv
from a2b_lightvpr.checkpointing import file_sha256,write_json

def prepare(ds):
    root=Path('data/generated/multidataset_qsattn')
    rows=list(csv.DictReader((root/'manifest.csv').open()))
    selected=[r for r in rows if r['dataset']==ds]
    source=list(csv.DictReader(open(f'data/manifests_multidataset_qsattn/{ds}_vpr_train.csv')))
    expected={r['image_id']:r for r in source}
    checkpoint=Path(f'runs/generator/qsattn_{ds}_30000_seed0/generator_final.pt')
    sha=file_sha256(checkpoint)
    meta=json.loads((root/ds/'translation_metadata.json').read_text())
    if meta['checkpoint_sha256']!=sha or meta['checkpoint_step']!=30000:raise ValueError('Wrong generator')
    seen=set()
    for r in selected:
        k=r['image_id']
        if k in seen or k not in expected or r['split']!='train' or r['generator_sha256']!=sha:raise ValueError('Invalid selected row')
        if Path(r['source_path']).resolve()!=Path(expected[k]['source_path']).resolve() or r['place_id']!=expected[k]['place_id']:raise ValueError('Source mismatch')
        if not Path(r['generated_path']).resolve().is_relative_to((root/ds).resolve()):raise ValueError('Wrong generator directory')
        for field in ['source_path','generated_path']:
            if not Path(r[field]).is_file():raise FileNotFoundError(r[field])
        seen.add(k)
    if seen!=set(expected):raise ValueError('Incomplete translated dataset')
    dest=Path(f'data/generated/{ds}_only_qsattn30k/manifest.csv')
    write_csv(dest,selected)
    write_json(dict(dataset=ds,rows=len(selected),generator_sha256=sha,manifest_sha256=file_sha256(dest),source_manifest_sha256=file_sha256(root/'manifest.csv')),dest.with_suffix('.json'))
    print(dest,len(selected),'validated rows')

def summarize(ds):
    root=Path(f'runs/vpr/{ds}_only_512')
    protocols=['svox_night','svox_day','svox_sun'] if ds=='svox' else ['msls_night','msls_day_val']
    models={}
    for group in ['day_only','day_fake','cdnd']:
        p=root/f'{group}_seed0/final_suite/summary.json'
        d=json.loads(p.read_text())
        models[group]={'source':str(p),'checkpoint':d.get('checkpoint'),
                       'input_size':d.get('input_size'),'datasets':{k:d['datasets'][k] for k in protocols}}
    delta={}
    for a,b in [('day_only','day_fake'),('day_fake','cdnd')]:
        delta[f'{b}_minus_{a}']={k:{r:100*(models[b]['datasets'][k]['recalls'][r]-models[a]['datasets'][k]['recalls'][r])
                                      for r in ['1','5','10']} for k in protocols}
    write_json({'selection_rule':'fixed_budget_last','models':models,'delta_percentage_points':delta},root/'ablation_summary.json')
    with (root/'ablation_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Method','Protocol','R@1 %','R@5 %','R@10 %'])
        for group,m in models.items():
            for k in protocols:w.writerow([group,k,*[100*m['datasets'][k]['recalls'][r] for r in ['1','5','10']]])
    print(root/'ablation_summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['svox','msls']);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    summarize(a.dataset) if a.summarize else prepare(a.dataset)
