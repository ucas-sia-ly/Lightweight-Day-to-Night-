"""Batch-one latency: encoder, image-to-descriptor, and synthetic-gallery search."""
import argparse
import csv
import time
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from a2b_lightvpr.checkpointing import write_json, file_sha256
from a2b_lightvpr.config import load_config
from a2b_lightvpr.qualitative.export import _models
from a2b_lightvpr.references.models import FixedResizeTransform

ORDER = ('cosplace', 'official_mixvpr512', 'eigenplaces', 'nocplace',
         'anyloc_pca', 'crica_pca', 'salad', 'boq_pca', 'cdnd')
LABELS = ('CosPlace', 'MixVPR', 'EigenPlaces', 'NocPlace', 'AnyLoc-PCA',
          'CricaVPR-PCA', 'SALAD', 'BoQ-PCA', 'CDND (Ours)')


def measure(operation, warmup, iterations):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return dict(mean_ms=float(np.mean(values)), p50_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), samples_ms=values)


def plot(results, path, gallery_size):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [r['method'] for r in results]
    inference = [r['encoder']['mean_ms'] for r in results]
    search = [r['search']['mean_ms'] for r in results]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    ax.bar(x, inference, color='#b6dfe7', label='Descriptor extraction (includes PCA)')
    ax.bar(x, search, bottom=inference, color='#e8b9b3', label='Similarity + top-1')
    for i, (a, b) in enumerate(zip(inference, search)):
        ax.annotate(f'{a+b:.2f}', (i, a+b), xytext=(0, 4), textcoords='offset points', ha='center')
    ax.set_xticks(x, names, rotation=25, ha='right')
    ax.set_ylabel('Mean latency (ms/query)')
    ax.set_title(f'Batch=1; synthetic gallery N={gallery_size:,}; sum of separately measured stages')
    ax.set_ylim(0, max(a+b for a,b in zip(inference, search))*1.3)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path / 'latency.png', dpi=240)
    fig.savefig(path / 'latency.pdf')
    plt.close(fig)


def gpu_snapshot(require_idle=False):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    processes = list(csv.reader(result.stdout.splitlines()))
    competing = [r for r in processes if len(r) >= 2 and r[0].strip() != str(os.getpid())
                 and 'python' in r[1].lower()]
    if require_idle and competing:
        raise RuntimeError(f'Other Python GPU workloads detected; wait for them to finish: {competing}')
    hardware = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version,temperature.gpu,utilization.gpu,memory.used',
                               '--format=csv,noheader'], capture_output=True, text=True, check=True)
    return {'utc': datetime.now(timezone.utc).isoformat(), 'processes': processes,
            'hardware': hardware.stdout.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vpr/cdnd_salad_512.yaml')
    parser.add_argument('--image', help='Optional shared real image; otherwise fixed synthetic 640x480 RGB')
    parser.add_argument('--output', default='runs/benchmarks/nine_methods_continuous')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--gallery-size', type=int, default=10000)
    parser.add_argument('--precision', choices=('fp16','fp32'), default='fp16')
    parser.add_argument('--models', nargs='+', choices=ORDER, default=list(ORDER))
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.gallery_size) < 1:
        parser.error('warmup, iterations and gallery-size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('This comparison requires a CUDA GPU')
    initial_gpu = gpu_snapshot(require_idle=True)
    import matplotlib  # Fail before measurement if plotting dependency is absent.
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(args.output)
    if (output / 'results.json').exists():
        raise FileExistsError(f'Use a new --output directory; existing run preserved: {output}')
    output.mkdir(parents=True, exist_ok=True)
    if args.image:
        with Image.open(args.image) as img:
            image = img.convert('RGB')
        image_info = {'path':str(Path(args.image).resolve()), 'sha256':file_sha256(args.image)}
    else:
        image = Image.fromarray(np.random.default_rng(0).integers(0,256,(480,640,3),dtype=np.uint8))
        image_info = {'kind':'synthetic RGB, seed=0, 640x480'}
    payload = {
        'complete': False, 'initial_gpu': initial_gpu,
        'timing_script_sha256': file_sha256(__file__),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cpu_threads': torch.get_num_threads(),
        'gpu':torch.cuda.get_device_name(0), 'torch':str(torch.__version__),
        'batch_size':1, 'precision':args.precision, 'warmup':args.warmup,
        'iterations':args.iterations, 'image':image_info,
        'gallery_size':args.gallery_size, 'gallery':'synthetic normalized FP32 GPU descriptors',
        'search_definition':'FP32 cosine matrix-vector product + argmax, TF32 disabled; database already on GPU',
        'encoder_definition':'GPU-resident input to normalized descriptor, including PCA; excludes decode/resize/transfer',
        'image_pipeline_definition':'decoded CPU PIL image -> resize/normalize -> H2D -> descriptor; excludes disk/decode',
        'notes':'NocPlace is the user-approved clean5k reproduction. CDND uses original seed0, not new GSV experiment.',
        'results':[],
    }
    config = load_config(args.config)
    with torch.inference_mode():
        for name, model, transform, checkpoint in _models(config, tuple(args.models)):
            before_gpu = gpu_snapshot(require_idle=True)
            label = LABELS[ORDER.index(name)]
            # Match the latest square-input recall experiment, not old qualitative sizes.
            if name == 'salad':
                transform = FixedResizeTransform(322, 322)
            elif name == 'nocplace':
                transform = FixedResizeTransform(320, 320)
            model.eval().cuda()
            sample = transform(image).unsqueeze(0).cuda()
            def encode():
                with torch.autocast('cuda', enabled=args.precision=='fp16'):
                    return model.encode(sample)
            descriptor = encode()
            if descriptor.shape != (1, model.descriptor_dim) or not torch.isfinite(descriptor).all():
                raise RuntimeError(f'{label}: invalid descriptor')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            encoder = measure(encode, args.warmup, args.iterations)
            peak = torch.cuda.max_memory_allocated()/1024**2
            def pipeline():
                tensor = transform(image).unsqueeze(0).cuda()
                with torch.autocast('cuda', enabled=args.precision=='fp16'):
                    return model.encode(tensor)
            image_pipeline = measure(pipeline, args.warmup, args.iterations)
            generator = torch.Generator(device='cuda').manual_seed(0)
            gallery = F.normalize(torch.randn(args.gallery_size, model.descriptor_dim,
                                              generator=generator, device='cuda'), dim=1)
            query = F.normalize(descriptor.float(), dim=1)[0]
            search = measure(lambda: torch.argmax(gallery @ query), args.warmup, args.iterations)
            result = dict(model_id=name, method=label, input_size=list(sample.shape[-2:]),
                          transform=type(transform).__name__, descriptor_dim=model.descriptor_dim,
                          parameters=sum(p.numel() for p in model.parameters()),
                          checkpoint=str(checkpoint.resolve()), artifacts=model.qualitative_artifacts,
                          peak_encoder_cuda_mib=peak, encoder=encoder, image_pipeline=image_pipeline, search=search)
            result['gpu_before'] = before_gpu
            result['gpu_after'] = gpu_snapshot(require_idle=True)
            payload['results'].append(result)
            write_json(payload, output/'results.json')
            print(f'{label}: encoder p50={encoder["p50_ms"]:.3f}ms, p95={encoder["p95_ms"]:.3f}ms; '
                  f'with preprocessing p50={image_pipeline["p50_ms"]:.3f}ms; search p50={search["p50_ms"]:.3f}ms', flush=True)
            model.cpu()
            del model, sample, descriptor, gallery, query
            torch.cuda.empty_cache()
    with (output/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['Method','Input HxW','Dim','Params M','Encoder mean ms','Encoder p50 ms','Encoder p95 ms',
                         'Image pipeline p50 ms','Search mean ms','Search p50 ms','Search p95 ms'])
        for r in payload['results']:
            writer.writerow([r['method'],'x'.join(map(str,r['input_size'])),r['descriptor_dim'],r['parameters']/1e6,
                r['encoder']['mean_ms'],r['encoder']['p50_ms'],r['encoder']['p95_ms'],r['image_pipeline']['p50_ms'],
                r['search']['mean_ms'],r['search']['p50_ms'],r['search']['p95_ms']])
    with (output/'samples.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Stage', 'Iteration', 'Latency ms'])
        for r in payload['results']:
            for stage in ('encoder', 'image_pipeline', 'search'):
                for i, value in enumerate(r[stage]['samples_ms']):
                    writer.writerow([r['method'], stage, i + 1, value])
    plot(payload['results'], output, args.gallery_size)
    payload['complete'] = True
    payload['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(payload, output/'results.json')
    print(f'Complete: {output}', flush=True)


if __name__ == '__main__':
    main()
