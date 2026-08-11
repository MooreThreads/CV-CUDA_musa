#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parent


def fnum(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def case_key(row):
    return (row['Benchmark'], row['InOutDataType'], row['shape'], row['params'])


def parse_args():
    parser = argparse.ArgumentParser(description='Generate OpenCV CPU vs CV-CUDA backend comparison tables.')
    parser.add_argument('--raw', type=Path, default=DEFAULT_ROOT / 'results' / 'backend_compare' / 'opencv_cvcuda_backend_raw.csv')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_ROOT / 'results' / 'backend_compare')
    parser.add_argument('--benchmark-source', type=Path, default=DEFAULT_ROOT / 'opencv_cvcuda_benchmark_compare.cpp')
    parser.add_argument('--opencv-source', type=Path, default=DEFAULT_ROOT / 'opencv')
    parser.add_argument('--opencv-branch', default='5.x')
    parser.add_argument('--cvcuda-root', type=Path, default=Path('/home/mt/data/yaguang/CV-CUDA'))
    parser.add_argument('--cvcuda-build', type=Path, default=Path('/home/mt/data/yaguang/CV-CUDA/build-cuda-bench-compare'))
    parser.add_argument('--backend', choices=['MUSA', 'CUDA'], default='MUSA')
    return parser.parse_args()


def load_rows(raw_path):
    opencv = {}
    cvcuda = {}
    with raw_path.open(newline='') as f:
        for row in csv.DictReader(f):
            if row.get('Status') != 'success':
                continue
            backend = row['Backend']
            if backend == 'opencv_cpu':
                opencv[case_key(row)] = row
            elif backend == 'cvcuda_gpu' or backend.startswith('cvcuda_'):
                cvcuda[case_key(row)] = row
    return opencv, cvcuda


def make_rows(opencv, cvcuda):
    rows = []
    for key in sorted(set(opencv) & set(cvcuda)):
        o = opencv[key]
        c = cvcuda[key]
        opencv_sec = fnum(o['CPU Time (sec)'])
        cvcuda_gpu_sec = fnum(c['GPU Time (sec)'])
        cvcuda_cpu_sec = fnum(c['CPU Time (sec)'])
        rows.append({
            'Benchmark': key[0],
            'InOutDataType': key[1],
            'shape': key[2],
            'params': key[3],
            'OpenCV backend': o['Backend'],
            'CV-CUDA backend': c['Backend'],
            'CV-CUDA device': c.get('Device', ''),
            'OpenCV CPU ms': opencv_sec * 1000,
            'CV-CUDA GPU ms': cvcuda_gpu_sec * 1000,
            'CV-CUDA CPU submit ms': cvcuda_cpu_sec * 1000,
            'Speedup OpenCV_CPU / CVCUDA_GPU': opencv_sec / cvcuda_gpu_sec if cvcuda_gpu_sec > 0 else math.nan,
            'OpenCV samples': o['Samples'],
            'CV-CUDA samples': c['Samples'],
            'Notes': 'Same C++ benchmark case dispatched through OpenCV CPU backend and direct CV-CUDA API backend',
        })
    return rows


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else [
        'Benchmark', 'InOutDataType', 'shape', 'params', 'OpenCV backend', 'CV-CUDA backend',
        'CV-CUDA device', 'OpenCV CPU ms', 'CV-CUDA GPU ms', 'CV-CUDA CPU submit ms',
        'Speedup OpenCV_CPU / CVCUDA_GPU', 'OpenCV samples', 'CV-CUDA samples', 'Notes'
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, args, rows):
    with path.open('w') as f:
        f.write('# OpenCV/CV-CUDA Benchmark Compare\n\n')
        f.write(f'- Benchmark harness: `{args.benchmark_source}`\n')
        f.write(f'- Raw CSV: `{args.raw}`\n')
        f.write(f'- Selected backend: `{args.backend}`\n')
        f.write(f'- OpenCV source: `{args.opencv_source}`, branch `{args.opencv_branch}`\n')
        f.write(f'- CV-CUDA source/API: `{args.cvcuda_root}`\n')
        f.write(f'- CV-CUDA build: `{args.cvcuda_build}`\n')
        f.write('- Integration mode: one benchmark case list, two backends: `opencv_cpu` and direct `cvcuda_gpu` API calls\n')
        f.write('- Input compared: `1x1080x1920`, data types U8/F32 depending on operator support\n')
        f.write('- Timing: OpenCV CPU wall time; CV-CUDA GPU event time plus CPU submit/sync wall time\n\n')
        if rows:
            headers = [
                'Benchmark', 'InOutDataType', 'shape', 'params', 'OpenCV CPU ms',
                f'CV-CUDA {args.backend} GPU ms', f'CV-CUDA {args.backend} CPU submit ms',
                'Speedup OpenCV_CPU / CVCUDA_GPU'
            ]
            f.write('| ' + ' | '.join(headers) + ' |\n')
            f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
            for r in rows:
                f.write('| ' + ' | '.join([
                    str(r['Benchmark']),
                    str(r['InOutDataType']),
                    str(r['shape']),
                    str(r['params']),
                    f"{r['OpenCV CPU ms']:.6f}",
                    f"{r['CV-CUDA GPU ms']:.6f}",
                    f"{r['CV-CUDA CPU submit ms']:.6f}",
                    f"{r['Speedup OpenCV_CPU / CVCUDA_GPU']:.2f}x",
                ]) + ' |\n')
        else:
            f.write('No comparable rows found.\n')


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    opencv, cvcuda = load_rows(args.raw)
    rows = make_rows(opencv, cvcuda)

    csv_path = args.out_dir / 'opencv_backend_vs_cvcuda_backend.csv'
    md_path = args.out_dir / 'opencv_backend_vs_cvcuda_backend.md'
    write_csv(csv_path, rows)
    write_markdown(md_path, args, rows)

    print(csv_path)
    print(md_path)


if __name__ == '__main__':
    main()
