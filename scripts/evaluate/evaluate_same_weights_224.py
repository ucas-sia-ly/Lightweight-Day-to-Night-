"""224-input resolution ablation, pinned to the nine-method night table artifacts."""
import argparse
import csv
import json
from pathlib import Path
import torch
from a2b_lightvpr.config import load_config
from a2b_lightvpr.checkpointing import file_sha256,write_json
from a2b_lightvpr.qualitative.export import _models
from a2b_lightvpr.references.models import FixedResizeTransform,suite_rows,query_protocol_fingerprint,evaluate_dual_encoder
from a2b_lightvpr.transforms import VPRTransform
from a2b_lightvpr.trained_suite import SUITE_PROTOCOLS

ANCHORS={
 'cosplace':'cosplace-r50-512-320/final_suite/native/adapted',
 'official_mixvpr512':'mixvpr-r50-512/final_suite/fitpad_320x320/adapted',
 'eigenplaces':'eigenplaces-r50-512-320/final_suite/native/adapted',
 'nocplace':'nocplace-r50-clean5k-resize320/final_suite/native/adapted',
 'anyloc_pca':'anyloc-dinov2-s-gsv-pca512/final_suite/native/adapted',
 'crica_pca':'cricavpr-dinov2-s-gsv322-pca512/final_suite/native/adapted',
 'salad':'salad-dinov2-s-gsv-eval322/final_suite/native/adapted',
 'boq_pca':'boq-dinov2-s-pca512/final_suite/native/adapted',
 'cdnd':None,
}
PROTOCOLS=('robotcar_project','svox_night','tokyo_night','msls_night')


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--smoke',action='store_true',help='Load all models and validate 224 forward only')
 p.add_argument('--datasets',nargs='+',choices=list(SUITE_PROTOCOLS),default=list(PROTOCOLS))
 p.add_argument('--output',default='runs/reference/same_weights_224')
 args=p.parse_args();torch.set_num_threads(4)
 config=load_config('configs/vpr/cdnd_salad_512.yaml')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 table=[]
 for name,model,_,checkpoint in _models(config,tuple(ANCHORS)):
  anchor=Path('runs/reference')/ANCHORS[name] if ANCHORS[name] else Path('runs/vpr/cdnd_512/salad_no_gate_seed0/final_suite')
  summary=json.loads((anchor/'summary.json').read_text())
  checksum=file_sha256(checkpoint)
  stored=summary.get('query_checkpoint_sha256',summary.get('checkpoint_sha256'))
  if stored is None and name=='cdnd':
   stored=json.loads((anchor/'robotcar_project/metrics.json').read_text())['checkpoint_sha256']
  if checksum!=stored:raise ValueError(f'{name}: checkpoint differs from original table')
  spec=model.qualitative_spec
  if spec.get('projection_checkpoint'):
   if file_sha256(spec['projection_checkpoint'])!=summary['projection_checkpoint_sha256']:raise ValueError(f'{name}: PCA differs')
  for path,value in summary.get('required_artifact_sha256',{}).items():
   if file_sha256(path)!=value:raise ValueError(f'{name}: artifact differs: {path}')
  transform=VPRTransform(224,224) if name in ('official_mixvpr512','cdnd') else FixedResizeTransform(224,224)
  model.eval().to(device)
  with torch.inference_mode():out=model.encode(torch.randn(1,3,224,224,device=device))
  assert out.shape==(1,summary['descriptor_dim']) and torch.isfinite(out).all()
  print(name,tuple(out.shape),'checkpoint/PCA verified',flush=True)
  if not args.smoke:
   root=Path(args.output)/name;root.mkdir(parents=True,exist_ok=True)
   for protocol in args.datasets:
    dataset,split,radius,yaw,_=SUITE_PROTOCOLS[protocol]
    db,queries=suite_rows(config,dataset,split)
    old_path=anchor/protocol/'metrics.json'
    old=json.loads(old_path.read_text()) if old_path.exists() else {}
    fingerprint=query_protocol_fingerprint(db,queries,radius,yaw)
    rebaseline=bool(old.get('query_protocol_fingerprint') and old['query_protocol_fingerprint']!=fingerprint)
    batch=1 if name=='crica_pca' else int(old.get('extraction_batch_size',16))
    identity=dict(checkpoint_sha256=checksum,artifacts=model.qualitative_artifacts,
        input_size=[224,224],descriptor_dim=model.descriptor_dim,transform=type(transform).__name__,
        batch_size=batch,query_protocol_fingerprint=query_protocol_fingerprint(db,queries,radius,yaw),
        implementation_sha256=file_sha256(__file__),precision='fp32',torch=str(torch.__version__))
    baseline_path=None
    if rebaseline:
     original_size=old.get('input_size',summary['input_size'])
     baseline_transform=VPRTransform(*original_size) if name in ('official_mixvpr512','cdnd') else FixedResizeTransform(*original_size)
     baseline_identity={**identity,'input_size':original_size,'transform':type(baseline_transform).__name__}
     baseline_path=root/f'{protocol}_original_size_retest.json'
     previous=json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
     if previous.get('identity')!=baseline_identity:
      print(f'{name}/{protocol}: historical fingerprint mismatch; retesting original size {original_size} on current rows',flush=True)
      baseline_metrics=evaluate_dual_encoder(model,model,db,queries,baseline_transform,radius,yaw,(1,5,10),device,batch,4)
      write_json(dict(identity=baseline_identity,metrics=baseline_metrics,anchor=str(anchor),
          reason='Historical row identity differs; original-size and 224 now share current rows',
          historical_query_protocol_fingerprint=old['query_protocol_fingerprint'],
          historical_recalls=old.get('recalls')),baseline_path)
    path=root/f'{protocol}.json' 
    saved=json.loads(path.read_text()) if path.exists() else {}
    if saved.get('identity')==identity:
     metrics=saved['metrics']
    else:
     metrics=evaluate_dual_encoder(model,model,db,queries,transform,radius,yaw,(1,5,10),device,batch,4)
     write_json(dict(identity=identity,metrics=metrics,anchor=str(anchor),original_size_retest=str(baseline_path) if baseline_path else None),path)
    table.append(dict(Method=name,Dataset=protocol,Image='224x224',Dim=model.descriptor_dim,Params_M=sum(p.numel() for p in model.parameters())/1e6,
        **{f'R@{k}':100*metrics['recalls'][str(k)] for k in (1,5,10)}))
  model.cpu();del model,out;torch.cuda.empty_cache()
 if table:
  with (Path(args.output)/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)

if __name__=='__main__':main()
