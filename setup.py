# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
defines = [
    "-DTORCH_STABLE_ONLY",
    "-DUSE_CUDA",
    "-DPy_LIMITED_API=0x030A0000",
    "-DTORCH_TARGET_VERSION=0x020A000000000000",
]


def extension(name, sources, *, fast_math=False):
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        *defines,
    ]
    if fast_math:
        nvcc_flags.append("--use_fast_math")
    return CUDAExtension(
        name,
        sources,
        include_dirs=[
            str(Path("third_party/cutlass/include").resolve()),
            str(Path("third_party/ninfer/src").resolve()),
            str(Path("third_party/flash_attention/csrc/flash_attn/src").resolve()),
        ],
        py_limited_api=True,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", *defines],
            "nvcc": nvcc_flags,
        },
    )


setup(
    ext_modules=[
        extension(
            "consumer_int8._gguf",
            ["csrc/gguf_bindings.cpp", "csrc/gguf_q8.cu", "csrc/gguf_q8_packed.cu"],
        ),
        extension(
            "consumer_int8._C",
            [
                "csrc/bindings.cpp",
                "csrc/kernels.cu",
                "csrc/silu_quant.cu",
                "csrc/qwen_aux.cu",
                "csrc/gdn_conv.cu",
                "csrc/cutlass_mm.cu",
                "csrc/gdn_prefill.cu",
                "csrc/gemv.cu",
                "csrc/allspark_layout.cu",
                "csrc/paged_attention.cu",
                "csrc/paged_decode.cu",
                "csrc/kv_unpack.cu",
                "csrc/fused_mm.cu",
            ],
        ),
        extension(
            "consumer_int8._fa2",
            ["csrc/fa2_bindings.cpp", "csrc/fa2_prefill.cu"],
            fast_math=True,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
