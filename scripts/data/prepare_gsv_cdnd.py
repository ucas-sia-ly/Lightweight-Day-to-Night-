"""Prepare a deterministic, city-qualified GSV manifest for the existing CDND pipeline."""
import argparse
import csv
from pathlib import Path
from a2b_lightvpr.references.training import gsv_rows, gsv_manifest_fingerprint
from a2b_lightvpr.checkpointing import write_json, file_sha256


def prepare(root, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.csv')
    fields = ['image_id', 'source_path', 'split', 'domain', 'dataset', 'place_id', 'role']
    count = 0
    places = gsv_rows(root)
    try:
        with temporary.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for place in places:
                for path in place['paths']:
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
                    writer.writerow(dict(image_id=Path(path).stem, source_path=path,
                        split='train', domain='day', dataset='gsv_cities',
                        place_id=f"{place['city']}:{place['place_id']}", role='database'))
                    count += 1
        if output.exists():
            if file_sha256(output) != file_sha256(temporary):
                raise RuntimeError('Existing GSV manifest differs; use a new experiment directory')
        else:
            temporary.replace(output)
        write_json(dict(gsv_root=str(Path(root).resolve()), places=len(places), images=count,
                        gsv_manifest_sha256=gsv_manifest_fingerprint(root),
                        manifest_sha256=file_sha256(output)), output.with_suffix('.json'))
    finally:
        temporary.unlink(missing_ok=True)
    print(f'{output}: {len(places)} places, {count} images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/home/admin123/github/WORKSPACE/dataset/gsv-cities')
    parser.add_argument('--output', default='data/manifests_gsv_cdnd/vpr_train.csv')
    args = parser.parse_args()
    prepare(args.root, args.output)
