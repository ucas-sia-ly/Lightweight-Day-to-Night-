import importlib.util,csv,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('merge',str(Path.cwd()/'scripts/data/prepare_multidataset_qsattn.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def setup(tmp):
 root=tmp/'generated';src=tmp/'source'
 for ds in m.DATASETS:
  d=root/ds;d.mkdir(parents=True)
  ck=d/'generator.pt';ck.write_bytes(ds.encode());img=d/'fake.png';img.write_bytes(b'fake');day=d/'day.png';day.write_bytes(b'day')
  row=dict(dataset=ds,image_id=ds,source_path=str(day),generated_path=str(img),place_id='p',split='train',generator_sha256=m.file_sha256(ck))
  m.write_csv(src/f'{ds}_vpr_train.csv',[row])
  m.write_csv(d/'manifest.csv',[row,{**row,'dataset':'foreign','image_id':'foreign'}])
  (d/'translation_metadata.json').write_text(json.dumps(dict(checkpoint=str(ck),checkpoint_sha256=m.file_sha256(ck),checkpoint_step=30000,smoke_limit=None)))
 return root,src

def test_filters_by_generator_dataset_and_is_repeatable(tmp_path):
 root,src=setup(tmp_path);m.merge_manifests(root,source_root=src)
 p=root/'manifest.csv';first=p.read_bytes();assert len(list(csv.DictReader(p.open())))==3
 m.merge_manifests(root,source_root=src);assert p.read_bytes()==first

def test_rejects_duplicate_before_changing_existing_output(tmp_path):
 root,src=setup(tmp_path);m.merge_manifests(root,source_root=src)
 p=root/'manifest.csv';first=p.read_bytes();q=root/'svox/manifest.csv';rows=list(csv.DictReader(q.open()));m.write_csv(q,rows+[rows[0]])
 with pytest.raises(ValueError,match='Duplicate'):m.merge_manifests(root,source_root=src)
 assert p.read_bytes()==first

def test_rejects_wrong_generator(tmp_path):
 root,src=setup(tmp_path);q=root/'msls/manifest.csv';rows=list(csv.DictReader(q.open()));rows[0]['generator_sha256']='wrong';m.write_csv(q,rows)
 with pytest.raises(ValueError,match='wrong generator'):m.merge_manifests(root,source_root=src)
