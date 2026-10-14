"""Prepare per-dataset QS-Attn inputs; merge only each generator's own dataset."""
import argparse
import csv
import io
import json
import shutil
from pathlib import Path

from a2b_lightvpr.checkpointing import write_json, file_sha256
from a2b_lightvpr.paths import resolve_manifest_image_path, resolve_resource_path

DATASETS = ('robotcar', 'svox', 'msls')


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        raise ValueError(f'Empty manifest: {path}')
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    content = buf.getvalue().encode()
    if path.exists() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(path.name + '.' + file_sha256(path)[:12] + '.bak')
        if not backup.exists():
            shutil.copy2(path, backup)
    temporary = path.with_suffix('.tmp.csv')
    temporary.write_bytes(content)
    temporary.replace(path)


def filter_manifests(source='data/manifests_clean_night', out='data/manifests_multidataset_qsattn'):
    out = Path(out)
    for name in ('generator_train', 'vpr_train'):
        with (Path(source) / f'{name}.csv').open() as f:
            original = list(csv.DictReader(f))
        combined = []
        for ds in DATASETS:
            rows = [r for r in original if r.get('dataset') == ds and r.get('split') == 'train'
                    and r.get('valid', '1').lower() not in ('0', 'false', 'no')]
            write_csv(out / f'{ds}_{name}.csv', rows)
            combined.extend(rows)
            print(ds, name, len(rows))
        write_csv(out / f'combined_{name}.csv', combined)


def merge_manifests(root='data/generated/multidataset_qsattn', out=None,
                    source_root='data/manifests_multidataset_qsattn'):
    root = Path(root)
    out = Path(out or root / 'manifest.csv')
    allrows, seen, audit = [], set(), {}
    for ds in DATASETS:
        manifest = root / ds / 'manifest.csv'
        metadata_path = manifest.with_name('translation_metadata.json')
        metadata = json.loads(metadata_path.read_text())
        checkpoint = resolve_resource_path(metadata['checkpoint'])
        generator_sha = file_sha256(checkpoint)
        if metadata['checkpoint_sha256'] != generator_sha or metadata['checkpoint_step'] != 30000 or metadata.get('smoke_limit') is not None:
            raise ValueError(f'{ds}: wrong or incomplete translation checkpoint')
        source_manifest = Path(source_root) / f'{ds}_vpr_train.csv'
        with source_manifest.open() as f:
            source = list(csv.DictReader(f))
        expected = {r['image_id']: r for r in source}
        if len(expected) != len(source):
            raise ValueError(f'{ds}: source image IDs are not unique')
        with manifest.open() as f:
            original = list(csv.DictReader(f))
        # Legacy manifests accidentally translated all three datasets. Select by
        # dataset AND generator provenance; never deduplicate by taking first row.
        selected = [r for r in original if r['dataset'] == ds]
        local_ids = set()
        for r in selected:
            key = r['image_id']
            if key in seen or key in local_ids:
                raise ValueError(f'Duplicate image_id in selected rows: {key}')
            if key not in expected or r['split'] != 'train':
                raise ValueError(f'{ds}: unexpected source row {key}')
            src = expected[key]
            source_image = Path(resolve_manifest_image_path(r['source_path'], manifest)).resolve()
            expected_image = Path(resolve_manifest_image_path(src['source_path'], source_manifest)).resolve()
            if source_image != expected_image or r['place_id'] != src['place_id']:
                raise ValueError(f'{ds}: source identity mismatch {key}')
            if r['generator_sha256'] != generator_sha:
                raise ValueError(f'{ds}: wrong generator for {key}')
            image = Path(resolve_manifest_image_path(r['generated_path'], manifest)).resolve()
            if not image.is_relative_to((root / ds).resolve()) or not image.is_file() or not source_image.is_file():
                raise ValueError(f'{ds}: missing or incorrectly routed image {key}')
            local_ids.add(key)
        if local_ids != set(expected):
            raise ValueError(f'{ds}: incomplete translation; expected {len(expected)}, found {len(local_ids)}')
        selected.sort(key=lambda r: r['image_id'])
        seen.update(local_ids)
        # Keep original path strings in outputs; normalization above only checks
        # migrated resources, preserving historical manifest bytes and hashes.
        allrows.extend(selected)
        audit[ds] = dict(selected=len(selected), excluded_other_datasets=len(original)-len(selected),
                         generator=str(checkpoint), generator_sha256=generator_sha,
                         input_manifest_sha256=file_sha256(manifest))
    # Validation finishes before replacing the combined manifest. Preserve old file.
    write_csv(out, allrows)
    write_json({'datasets': audit, 'rows': len(allrows), 'unique_image_ids':len(seen),
                'manifest_sha256':file_sha256(out)}, out.with_suffix('.json'))
    print(out, len(allrows), 'rows; all IDs unique; routing verified')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['filter', 'merge'])
    a = p.parse_args()
    filter_manifests() if a.stage == 'filter' else merge_manifests()
