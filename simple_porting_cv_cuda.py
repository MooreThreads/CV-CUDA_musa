#!/usr/bin/env python3
"""Round-1 CV-CUDA CUDA->MUSA source porting script.

Mapping provenance:
- Container JSON mappings under /usr/local/musa/tools and torch_musa/utils/mapping.
- Project-local folder/header rewrites are not CUDA API mappings; they implement the
  parallel-folder contract from CODE_ANALYSIS.md Required Component Closure.
"""
import json
import os
import re
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

JSONS = [
    "/usr/local/lib/python3.10/dist-packages/torch_musa/utils/mapping/general.json",
    "/usr/local/lib/python3.10/dist-packages/torch_musa/utils/mapping/dnn-naming.json",
    "/usr/local/lib/python3.10/dist-packages/torch_musa/utils/mapping/include.json",
    "/usr/local/lib/python3.10/dist-packages/torch_musa/utils/mapping/extra.json",
    "/usr/local/lib/python3.10/dist-packages/torch_musa/utils/mapping/dnn-header.json",
    "/usr/local/musa/tools/general.json",
    "/usr/local/musa/tools/ccl-naming.json",
    "/usr/local/musa/tools/dnn-naming.json",
    "/usr/local/musa/tools/include.json",
    "/usr/local/musa/tools/extra.json",
    "/usr/local/musa/tools/ccl-header.json",
    "/usr/local/musa/tools/dnn-header.json",
    "/usr/local/musa/tools/torch-naming.json",
]


def load_jsons():
    rules = {}
    for path in JSONS:
        p = Path(path)
        if p.exists():
            rules.update(json.loads(p.read_text()))
    return rules


def parse_rewrite_scopes():
    # CV-CUDA currently has no selected_deps rewrite_scope blocks, but keep the
    # required extraction hook for reproducible future reruns.
    additions, extra_dirs = {}, []
    text = (PROJECT_DIR / "CODE_ANALYSIS.md").read_text()
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    for block in re.finditer(r"rewrite_scope:\s*\n((?:\s{2,}.*\n)+)", text):
        if yaml is None:
            continue
        scope = yaml.safe_load("rewrite_scope:\n" + block.group(1))["rewrite_scope"]
        replacement = scope["replacement"]
        for hdr in scope.get("headers", []):
            prefix = hdr.split("*", 1)[0].rstrip("/")
            additions[f"#include <{prefix}/"] = f"#include <{replacement}/"
            additions[f"#include \"{prefix}/"] = f"#include \"{replacement}/"
        for ns in scope.get("namespaces", []):
            additions[f"{ns}::"] = f"{replacement}::"
            additions[f"::{ns}::"] = f"::{replacement}::"
        extra_dirs.extend(scope.get("target_dirs", []))
    return additions, extra_dirs


REWRITE_SCOPE_ADDITIONS, EXTRA_DIRS = parse_rewrite_scopes()

MAPPING_RULES = {
    **load_jsons(),
    **REWRITE_SCOPE_ADDITIONS,
    # Project-local parallel-folder rewrites from CODE_ANALYSIS.md Required Component Closure.
    "cvcuda/cuda_tools/": "cvcuda/cuda_tools_musa/",
    "cvcuda/cuda_tools": "cvcuda/cuda_tools_musa",
    "cuda_tools/": "cuda_tools_musa/",
    # Project-local CUDA header extension rewrite for generated sibling headers.
    ".cuh\"": ".muh\"",
    ".cuh>": ".muh>",
    # Custom torch_musa/PyTorch internal mappings from musifier guardrails.
    "#include <c10/cuda/CUDAStream.h>": '#include "torch_musa/csrc/core/MUSAStream.h"',
    "#include <c10/cuda/CUDAGuard.h>": '#include "torch_musa/csrc/core/MUSAGuard.h"',
    "#include <ATen/cuda/CUDAContext.h>": '#include "torch_musa/csrc/aten/musa/MUSAContext.h"',
    "#include <ATen/cuda/Atomic.cuh>": '#include "torch_musa/csrc/aten/musa/MUSAAtomic.muh"\nusing at::musa::gpuAtomicAdd;',
    "at::cuda::": "at::musa::",
    "c10::cuda::": "c10::musa::",
    "::c10::cuda::": "::c10::musa::",
    "OptionalCUDAGuard": "OptionalMUSAGuard",
    "CUDACachingAllocator": "MUSACachingAllocator",
    ".is_cuda()": ".is_privateuseone()",
    ", CUDA,": ", PrivateUse1,",
}

KEY_SUBSTRINGS = (
    "cuda", "CUDA", "cublas", "CUBLAS", "cusolver", "CUSOLVER",
    "curand", "CURAND", "cudnn", "CUDNN", "driver_types", "vector_types",
    "library_types", "__nv", "NV", "CU", "cvcuda/cuda_tools", "cuda_tools/",
)
FILTERED_RULES = {
    key: value for key, value in MAPPING_RULES.items()
    if any(token in key for token in KEY_SUBSTRINGS) or key.startswith("#include") or key.startswith(".") or key.startswith(", ")
}
SORTED_RULES = sorted(FILTERED_RULES.items(), key=lambda item: len(item[0]), reverse=True)

EXT_MAP = {".cu": ".mu", ".cuh": ".muh"}
TEXT_EXTS = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".inl", ".cmake", ".txt"}


def output_name(name: str) -> str:
    p = Path(name)
    return p.with_suffix(EXT_MAP.get(p.suffix, p.suffix)).name


def should_skip_mapping(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")


def transform_text(text: str) -> str:
    out_lines = []
    for line in text.splitlines(True):
        if should_skip_mapping(line):
            out_lines.append(line)
            continue
        if "cub/" not in line:
            for src, dst in SORTED_RULES:
                if src in line:
                    line = line.replace(src, dst)
        out_lines.append(line)
    return "".join(out_lines)


def port_dir(src_rel: str, dst_rel: str, ignore_parts=()):
    src = PROJECT_DIR / src_rel
    dst = PROJECT_DIR / dst_rel
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        rel = root_p.relative_to(src)
        if any(part in root_p.parts for part in ignore_parts):
            dirs[:] = []
            continue
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            in_path = root_p / filename
            out_path = out_dir / output_name(filename)
            if in_path.suffix in TEXT_EXTS:
                out_path.write_text(transform_text(in_path.read_text(errors="surrogateescape")), errors="surrogateescape")
            else:
                shutil.copy2(in_path, out_path)


def main():
    port_dir("src/cvcuda/include/cvcuda/cuda_tools", "src/cvcuda/include/cvcuda/cuda_tools_musa")
    port_dir("src/cvcuda/priv", "src/cvcuda/priv_musa")
    port_dir("src/cvcuda/util", "src/cvcuda/util_musa")
    for extra in EXTRA_DIRS:
        src = Path(extra)
        if src.exists():
            port_dir(str(src), str(src) + "_musa")
    print(f"Ported with {len(FILTERED_RULES)}/{len(MAPPING_RULES)} mapping rules; rewrite scopes={len(REWRITE_SCOPE_ADDITIONS)}")


if __name__ == "__main__":
    main()
