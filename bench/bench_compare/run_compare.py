#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = THIS_DIR


def run(cmd, env=None):
    print('+ ' + ' '.join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def capture(cmd):
    print('+ ' + ' '.join(str(x) for x in cmd), flush=True)
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='')
        lines.append(line)
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return ''.join(lines)


def default_cvcuda_root():
    candidates = []
    for base in [DEFAULT_ROOT, *DEFAULT_ROOT.parents]:
        candidates.append(base / 'CV-CUDA')

    for path in candidates:
        if path.exists():
            return path
    return DEFAULT_ROOT / 'CV-CUDA'


def default_cvcuda_build(cvcuda_root, backend):
    names = ['build_musa_bench_mp31', 'build_musa'] if backend == 'MUSA' else ['build-cuda-bench-compare', 'build-cuda']
    for name in names:
        path = cvcuda_root / name
        if path.exists():
            return path
    return cvcuda_root / names[0]


def default_opencv_dir(install_dir, build_dir):
    candidates = [
        install_dir / 'lib' / 'cmake' / 'opencv5',
        install_dir / 'lib64' / 'cmake' / 'opencv5',
        install_dir / 'lib' / 'cmake' / 'opencv4',
        install_dir / 'lib64' / 'cmake' / 'opencv4',
        install_dir,
        build_dir,
    ]
    for path in candidates:
        if (path / 'OpenCVConfig.cmake').exists():
            return path
    return install_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build OpenCV, build OpenCV/CV-CUDA benchmark compare, run benchmark, and generate tables.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--backend', choices=['MUSA', 'CUDA'], default='MUSA',
                        help='GPU backend. MUSA is the default backend.')
    parser.add_argument('--skip-opencv-build', action='store_true',
                        help='Do not build OpenCV. Use --opencv-dir or the auto-detected default OpenCV_DIR.')
    parser.add_argument('--opencv-dir', type=Path,
                        help='Existing OpenCV_DIR containing OpenCVConfig.cmake. Default: auto-detect from ./opencv-build-<backend> and ./opencv-build-<backend>-work.')
    parser.add_argument('--opencv-branch', default='5.x',
                        help='OpenCV git branch to fetch/build when no usable OpenCV_DIR already exists.')
    parser.add_argument('--opencv-repo-url', default='https://github.com/opencv/opencv.git',
                        help='OpenCV git repository URL.')
    parser.add_argument('--opencv-source-dir', type=Path,
                        help='OpenCV source directory. Default: ./opencv-src.')
    parser.add_argument('--opencv-build-dir', type=Path,
                        help='OpenCV CMake build directory. Default: ./opencv-build-<backend>-work.')
    parser.add_argument('--opencv-install-dir', type=Path,
                        help='OpenCV install directory. Default: ./opencv-build-<backend>.')
    parser.add_argument('--opencv-force-clean', action='store_true',
                        help='Remove the OpenCV CMake build directory before configuring OpenCV.')
    parser.add_argument('--opencv-skip-fetch', action='store_true',
                        help='Do not clone/fetch OpenCV; use the existing --opencv-source-dir.')
    parser.add_argument('--cvcuda-root', type=Path,
                        help='CV-CUDA source root. Default: search ./CV-CUDA, ../CV-CUDA, and higher parent directories.')
    parser.add_argument('--cvcuda-build', type=Path,
                        help='CV-CUDA build root. MUSA default: <cvcuda-root>/build_musa_bench_mp31 or build_musa. CUDA default: <cvcuda-root>/build-cuda-bench-compare or build-cuda.')
    parser.add_argument('--benchmark-build-dir', type=Path,
                        help='OpenCV/CV-CUDA benchmark compare CMake build directory. Default: ./build-<backend>.')
    parser.add_argument('--out-dir', type=Path,
                        help='Benchmark result output directory. Default: ./results/backend_compare_<backend>.')
    parser.add_argument('--samples', type=int, default=20,
                        help='Measured sample count per case.')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Warmup iteration count per case before measurement.')
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1,
                        help='Parallel build jobs.')
    args = parser.parse_args()

    backend_lower = args.backend.lower()
    args.opencv_source_dir = args.opencv_source_dir or DEFAULT_ROOT / 'opencv-src'
    args.opencv_build_dir = args.opencv_build_dir or DEFAULT_ROOT / f'opencv-build-{backend_lower}-work'
    args.opencv_install_dir = args.opencv_install_dir or DEFAULT_ROOT / f'opencv-build-{backend_lower}'
    args.opencv_dir = args.opencv_dir or default_opencv_dir(args.opencv_install_dir, args.opencv_build_dir)
    args.cvcuda_root = args.cvcuda_root or default_cvcuda_root()
    args.cvcuda_build = args.cvcuda_build or default_cvcuda_build(args.cvcuda_root, args.backend)
    args.benchmark_build_dir = args.benchmark_build_dir or THIS_DIR / f'build-{backend_lower}'
    args.out_dir = args.out_dir or DEFAULT_ROOT / 'results' / f'backend_compare_{backend_lower}'
    return args


def parse_opencv_dir(output):
    opencv_dir = None
    for line in output.splitlines():
        if line.startswith('OpenCV_DIR='):
            opencv_dir = Path(line.split('=', 1)[1].strip())
    if not opencv_dir:
        raise SystemExit('build_opencv.py did not print OpenCV_DIR=...')
    return opencv_dir


def build_opencv(args):
    if (args.opencv_dir / 'OpenCVConfig.cmake').exists():
        print(f'Use existing OpenCV_DIR={args.opencv_dir}')
        return args.opencv_dir

    if args.skip_opencv_build:
        raise SystemExit(f'OpenCVConfig.cmake not found in --opencv-dir: {args.opencv_dir}')

    cmd = [
        sys.executable,
        str(THIS_DIR / 'build_opencv.py'),
        '--repo-url', args.opencv_repo_url,
        '--branch', args.opencv_branch,
        '--source-dir', str(args.opencv_source_dir),
        '--build-dir', str(args.opencv_build_dir),
        '--install-dir', str(args.opencv_install_dir),
        '--jobs', str(args.jobs),
    ]
    if args.opencv_force_clean:
        cmd.append('--force-clean')
    if args.opencv_skip_fetch:
        cmd.append('--skip-fetch')
    return parse_opencv_dir(capture(cmd))


def lib_dirs_for(path):
    return [p for p in [path / 'lib', path / 'lib64'] if p.exists()]


def benchmark_env(args):
    lib_dirs = []
    lib_dirs.extend(lib_dirs_for(args.opencv_install_dir))
    lib_dirs.extend(lib_dirs_for(args.opencv_build_dir))
    lib_dirs.append(args.cvcuda_build / 'lib')
    if args.backend == 'MUSA':
        lib_dirs.append(Path('/usr/local/musa/lib'))
    else:
        for cuda_root in [Path('/usr/local/cuda'), Path('/usr/local/cuda/lib64').parent]:
            lib_dirs.extend(lib_dirs_for(cuda_root))

    existing = os.environ.get('LD_LIBRARY_PATH')
    ld_parts = [str(p) for p in lib_dirs]
    if existing:
        ld_parts.append(existing)

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = ':'.join(ld_parts)
    return env


def main():
    args = parse_args()
    opencv_dir = build_opencv(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run([
        'cmake',
        '-S', str(THIS_DIR),
        '-B', str(args.benchmark_build_dir),
        f'-DBACKEND={args.backend}',
        f'-DOpenCV_DIR={opencv_dir}',
        f'-DCVCUDA_ROOT={args.cvcuda_root}',
        f'-DCVCUDA_BUILD={args.cvcuda_build}',
    ])
    run(['cmake', '--build', str(args.benchmark_build_dir), '--parallel', str(args.jobs), '--target', 'opencv_cvcuda_benchmark_compare'])

    raw_csv = args.out_dir / 'opencv_cvcuda_backend_raw.csv'
    run([
        str(args.benchmark_build_dir / 'opencv_cvcuda_benchmark_compare'),
        str(raw_csv),
        str(args.samples),
        str(args.warmup),
    ], env=benchmark_env(args))

    run([
        sys.executable,
        str(THIS_DIR / 'make_comparison.py'),
        '--raw', str(raw_csv),
        '--out-dir', str(args.out_dir),
        '--benchmark-source', str(THIS_DIR / 'opencv_cvcuda_benchmark_compare.cpp'),
        '--opencv-source', str(args.opencv_source_dir),
        '--opencv-branch', args.opencv_branch,
        '--cvcuda-root', str(args.cvcuda_root),
        '--cvcuda-build', str(args.cvcuda_build),
        '--backend', args.backend,
    ])

    print(f'Raw CSV: {raw_csv}')
    print(f'Comparison CSV: {args.out_dir / "opencv_backend_vs_cvcuda_backend.csv"}')
    print(f'Comparison Markdown: {args.out_dir / "opencv_backend_vs_cvcuda_backend.md"}')


if __name__ == '__main__':
    main()
