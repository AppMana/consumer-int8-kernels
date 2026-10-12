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
setup(
    ext_modules=[
        CUDAExtension(
            "consumer_int8._C",
            [
                "csrc/bindings.cpp",
                "csrc/kernels.cu",
                "csrc/cutlass_mm.cu",
                "csrc/gdn_prefill.cu",
                "csrc/gemv.cu",
                "csrc/allspark_layout.cu",
                "csrc/paged_attention.cu",
                "csrc/kv_unpack.cu",
            ],
            include_dirs=[
                str(Path("third_party/cutlass/include").resolve()),
                str(Path("third_party/ninfer/src").resolve()),
            ],
            py_limited_api=True,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", *defines],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                    *defines,
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
