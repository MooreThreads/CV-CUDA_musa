#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd, cwd=None):
    print('+ ' + ' '.join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch, build, and install OpenCV for opencv-cvcuda-compare.')
    parser.add_argument('--repo-url', default='https://github.com/opencv/opencv.git')
    parser.add_argument('--branch', default='5.x')
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--build-dir', required=True, type=Path)
    parser.add_argument('--install-dir', required=True, type=Path)
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--force-clean', action='store_true', help='Remove the build directory before configuring.')
    parser.add_argument('--skip-fetch', action='store_true', help='Do not clone/fetch OpenCV; use the existing source directory.')
    return parser.parse_args()


def update_source(args):
    source_dir = args.source_dir
    if args.skip_fetch:
        if not source_dir.exists():
            raise SystemExit(f'--skip-fetch was set but source dir does not exist: {source_dir}')
        return

    if not (source_dir / '.git').exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', '--branch', args.branch, '--depth', '1', args.repo_url, str(source_dir)])
        return

    run(['git', 'fetch', 'origin', args.branch], cwd=source_dir)
    run(['git', 'checkout', args.branch], cwd=source_dir)
    run(['git', 'pull', '--ff-only', 'origin', args.branch], cwd=source_dir)


def detect_opencv_dir(install_dir, build_dir):
    candidates = [
        install_dir / 'lib' / 'cmake' / 'opencv5',
        install_dir / 'lib64' / 'cmake' / 'opencv5',
        install_dir / 'lib' / 'cmake' / 'opencv4',
        install_dir / 'lib64' / 'cmake' / 'opencv4',
        build_dir,
    ]
    for path in candidates:
        if (path / 'OpenCVConfig.cmake').exists():
            return path
    searched = '\n'.join(str(p) for p in candidates)
    raise SystemExit(f'Could not find OpenCVConfig.cmake. Searched:\n{searched}')


def main():
    args = parse_args()
    update_source(args)

    if args.force_clean and args.build_dir.exists():
        shutil.rmtree(args.build_dir)

    args.build_dir.mkdir(parents=True, exist_ok=True)
    args.install_dir.mkdir(parents=True, exist_ok=True)

    run([
        'cmake',
        '-S', str(args.source_dir),
        '-B', str(args.build_dir),
        '-DCMAKE_BUILD_TYPE=Release',
        f'-DCMAKE_INSTALL_PREFIX={args.install_dir}',
        '-DBUILD_LIST=core,imgproc',
        '-DBUILD_TESTS=OFF',
        '-DBUILD_PERF_TESTS=OFF',
        '-DBUILD_EXAMPLES=OFF',
    ])
    run(['cmake', '--build', str(args.build_dir), '--parallel', str(args.jobs)])
    run(['cmake', '--install', str(args.build_dir)])

    opencv_dir = detect_opencv_dir(args.install_dir, args.build_dir)
    print(f'OpenCV_DIR={opencv_dir}')


if __name__ == '__main__':
    main()
