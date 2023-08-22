"""
@file   setup.py
@author Jianfei Guo, Shanghai AI Lab
@brief  nr3d_lib's pytorch/c++/CUDA extensions installation script.
"""

import os
import sys
import logging
import subprocess
from copy import deepcopy
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(SCRIPT_DIR, 'VERSION'),"r") as f: VERSION = f.read()

if torch.cuda.is_available():   
    if os.name == "nt":
        def find_cl_path():
            import glob
            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(glob.glob(
                    r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition),
                               reverse=True)
                if paths:
                    return paths[0]


        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ["PATH"] += ";" + cl_path

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    common_library_dirs = []
    # NOTE: On cluster machines's login node etc. where no GPU is installed, 
    #           `libcuda.so` will not be in usally place, 
    #           so `-lcuda` will raise "can not find -lcuda" error.
    #       To solve this, we can use `libcuda.so` in the `stubs` directory.
    # https://stackoverflow.com/questions/62999715/when-i-make-darknet-with-cuda-1-usr-bin-ld-cannot-find-lcudaoccured-how
    if '--fix-lcuda' in sys.argv:
        sys.argv.remove('--fix-lcuda')
        common_library_dirs.append(os.path.join(os.environ.get('CUDA_HOME'), 'lib64', 'stubs'))


    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor

    def get_cuda_bare_metal_version():
        raw_output = subprocess.check_output([os.path.join(CUDA_HOME, 'bin', 'nvcc'), "-V"], universal_newlines=True)
        output = raw_output.split()
        release_idx = output.index("release") + 1
        release = output[release_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]

        return raw_output, bare_metal_major, bare_metal_minor
else:
    raise EnvironmentError(
        "PyTorch CUDA is unavailable. nr3d_lib requires PyTorch to be installed with the CUDA backend.")

def get_ext_lotd():
    nvcc_flags = [
        "-std=c++14",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        # The following definitions must be undefined
        # since half-precision operation is required.
        '-U__CUDA_NO_HALF_OPERATORS__', 
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__', 
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    if os.name == "posix":
        c_flags = ["-std=c++14"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = ["/std:c++14"]

    print(f"Targeting compute capability {compute_capability}")

    definitions = [
    ]
    nvcc_flags += definitions
    c_flags += definitions

    # List of sources.
    csrc_dir = os.path.abspath(os.path.join(SCRIPT_DIR, 'csrc'))
    source_files = [
        os.path.join(csrc_dir, "lotd/src/lotd_impl_2d.cu"),
        os.path.join(csrc_dir, "lotd/src/lotd_impl_3d.cu"),
        os.path.join(csrc_dir, "lotd/src/lotd_impl_4d.cu"),
        os.path.join(csrc_dir, "lotd/src/lotd_torch_api.cu"),
        os.path.join(csrc_dir, "lotd/src/lotd.cpp"),
    ]

    libraries = []
    library_dirs = deepcopy(common_library_dirs)
    extra_objects = []

    ext = CUDAExtension(
        name="nr3d_lib_bindings._lotd",
        sources=source_files,
        include_dirs=[
            os.path.join(csrc_dir, "lotd", "include"), 
            # os.path.join(csrc_dir, "forest")
        ],
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
        libraries=libraries,
        library_dirs=library_dirs,
        extra_objects=extra_objects
    )
    return ext

def get_ext_pack_ops():
    nvcc_flags = [
        '-O3', '-std=c++14',
        # NOTE: We do not want built-in half ops. use cuda's half ops instead.
        #       https://discuss.pytorch.org/t/error-more-than-one-operator-matches-these-operands-in-thcnumerics-cuh/89935/2
        '-D__CUDA_NO_HALF_OPERATORS__', 
        '-D__CUDA_NO_HALF_CONVERSIONS__', 
        '-D__CUDA_NO_HALF2_OPERATORS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    
    if os.name == "posix":
        c_flags = ["-std=c++14"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = ["/std:c++14"]

    if '--use-thrust-sort' in sys.argv:
        sys.argv.remove('--use-thrust-sort')
        c_flags.append("-DPACK_OPS_USE_THRUST_SORT")
        nvcc_flags.append("-DPACK_OPS_USE_THRUST_SORT")

    library_dirs = deepcopy(common_library_dirs)

    csrc_dir = os.path.abspath(os.path.join(SCRIPT_DIR, 'csrc'))
    ext = CUDAExtension(
            name='nr3d_lib_bindings._pack_ops', # extension name, import this to use CUDA API
            sources=[os.path.join(csrc_dir, 'pack_ops', f) for f in [
                'pack_ops_cuda.cu',
                'pack_ops.cpp',
            ]],
            include_dirs=[
                os.path.join(csrc_dir, "pack_ops")
            ],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }, 
            library_dirs=library_dirs
    )
    return ext

def get_ext_occ_grid():
    nvcc_flags = [
        '-O3', '-std=c++14',
        # NOTE: We do not want built-in half ops. use cuda's half ops instead.
        #       https://discuss.pytorch.org/t/error-more-than-one-operator-matches-these-operands-in-thcnumerics-cuh/89935/2
        '-D__CUDA_NO_HALF_OPERATORS__', 
        '-D__CUDA_NO_HALF_CONVERSIONS__', 
        '-D__CUDA_NO_HALF2_OPERATORS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    
    if os.name == "posix":
        c_flags = ["-std=c++14"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = ["/std:c++14"]

    library_dirs = deepcopy(common_library_dirs)

    csrc_dir = os.path.abspath(os.path.join(SCRIPT_DIR, 'csrc'))
    ext = CUDAExtension(
            name='nr3d_lib_bindings._occ_grid', # extension name, import this to use CUDA API
            sources=[os.path.join(csrc_dir, 'occ_grid', f) for f in [
                'ray_marching.cu',
                'batched_marching.cu',
                # 'forest_marching.cu',
                'occ_grid.cpp',
            ]],
            include_dirs=[
                os.path.join(csrc_dir, "occ_grid", "include"), 
                # os.path.join(csrc_dir, "forest") # For forest occ grid marching
            ],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }, 
            library_dirs=library_dirs
    )
    return ext

def get_ext_spherical_embedder():
    # Modified from https://github.com/ashawkey/torch-ngp
    nvcc_flags = [
        '-O3', '-std=c++14',
        '-U__CUDA_NO_HALF_OPERATORS__', 
        '-U__CUDA_NO_HALF_CONVERSIONS__', 
        '-U__CUDA_NO_HALF2_OPERATORS__',
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    
    if os.name == "posix":
        c_flags = ["-std=c++14"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = ["/std:c++14"]
    
    library_dirs = deepcopy(common_library_dirs)
    
    csrc_dir = os.path.abspath(os.path.join(SCRIPT_DIR, 'csrc'))
    ext = CUDAExtension(
            name='nr3d_lib_bindings._shencoder', # extension name, import this to use CUDA API
            sources=[os.path.join(csrc_dir, 'shencoder', f) for f in [
                'shencoder.cu',
                'bindings.cpp',
            ]],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }, 
            library_dirs=library_dirs
    )
    return ext

def get_ext_frequency_embedder():
    # Modified from https://github.com/ashawkey/torch-ngp
    nvcc_flags = [
        '-O3', '-std=c++14',
        '-U__CUDA_NO_HALF_OPERATORS__', 
        '-U__CUDA_NO_HALF_CONVERSIONS__', 
        '-U__CUDA_NO_HALF2_OPERATORS__',
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    
    if os.name == "posix":
        c_flags = ["-std=c++14"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = ["/std:c++14"]
    
    library_dirs = deepcopy(common_library_dirs)
    
    csrc_dir = os.path.abspath(os.path.join(SCRIPT_DIR, 'csrc'))
    ext = CUDAExtension(
            name='nr3d_lib_bindings._freqencoder', # extension name, import this to use CUDA API
            sources=[os.path.join(csrc_dir, 'freqencoder', f) for f in [
                'freqencoder.cu',
                'bindings.cpp',
            ]],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }, 
            library_dirs=library_dirs
    )
    return ext

def get_ext_knn_from_pytorch3d():
    nvcc_flags = [
        "-std=c++14",
        
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]

    if os.name == "posix":
        c_flags = ["-std=c++14"]
        nvcc_flags += [
            '-Xcompiler="-mf16c"',
            '-Xcompiler="-Wno-float-conversion"',
            '-Xcompiler="-fno-strict-aliasing"',
            '-Xcompiler="-Wno-unused-parameter"', 
            "-Xcudafe=--diag_suppress=declared_but_not_referenced", 
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = ["/std:c++14"]

    library_dirs = deepcopy(common_library_dirs)
    csrc_dir = os.path.abspath(os.path.join(SCRIPT_DIR, 'csrc'))
    ext = CUDAExtension(
            name='nr3d_lib_bindings._knn_from_pytorch3d', # extension name, import this to use CUDA API
            sources=[os.path.join(csrc_dir, 'knn_from_pytorch3d', f) for f in [
                'knn.cu',
                'knn_cpu.cpp',
                'ext.cpp',
            ]],
            include_dirs=[
                os.path.join(csrc_dir, "knn_from_pytorch3d"), 
            ],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }, 
            define_macros=[("WITH_CUDA", None)], 
            library_dirs=library_dirs
    )

    return ext

def get_ext_spheretrace():
    nvcc_flags = [
        "-std=c++14",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        # The following definitions must be undefined
        # since half-precision operation is required.
        '-U__CUDA_NO_HALF_OPERATORS__', 
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__', 
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    if os.name == "posix":
        c_flags = ["-std=c++14"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=20012",
        ]
    elif os.name == "nt":
        c_flags = ["/std:c++14"]

    print(f"Targeting compute capability {compute_capability}")

    definitions = [
        f"-DTCNN_MIN_GPU_ARCH={compute_capability}",
        "-DNGP_OPTIX",
        "-O3",
        "-DNDEBUG"
    ]
    nvcc_flags += definitions
    c_flags += definitions

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    # List of sources.
    csrc_dir = os.path.abspath(os.path.join(SCRIPT_DIR, 'csrc'))
    source_files = [
        os.path.join(csrc_dir, "sphere_trace", "src", "entry.cu"),
        os.path.join(csrc_dir, "sphere_trace", "src", "sphere_tracer.cu")
    ]
    libraries = ["cuda"]
    library_dirs = deepcopy(common_library_dirs)
    extra_objects = []

    ext = CUDAExtension(
        name="nr3d_lib_bindings._sphere_trace",
        sources=source_files,
        include_dirs=[
            os.path.join(csrc_dir, "sphere_trace", "include"),
            os.path.join(csrc_dir, "third_party", "glm"),
        ],
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
        libraries=libraries,
        library_dirs=library_dirs,
        extra_objects=extra_objects
    )
    return ext

def get_extensions():
    ext_modules = []
    ext_modules.append(get_ext_pack_ops())
    # ext_modules.append(get_ext_forest())
    ext_modules.append(get_ext_occ_grid())
    ext_modules.append(get_ext_spherical_embedder())
    ext_modules.append(get_ext_frequency_embedder())
    ext_modules.append(get_ext_knn_from_pytorch3d())
    ext_modules.append(get_ext_lotd())
    ext_modules.append(get_ext_spheretrace())
    return ext_modules

setup(
    name="nr3d_lib",
    version=VERSION,
    description="nr3d_lib",
    long_description="nr3d_lib",
    keywords="nr3d_lib",
    url="https://github.com/PJLAB-ADG/nr3d_lib",
    author="Jianfei Guo, Nianchen Deng, Xinyang Li, Qiusheng Huang",
    author_email="guojianfei@pjlab.org.cn, dengnianchen@pjlab.org.cn, lixinyang@pjlab.org.cn, huangqiusheng@pjlab.org.cn",
    maintainer="Jianfei Guo",
    maintainer_email="guojianfei@pjlab.org.cn",
    download_url=f"https://github.com/PJLAB-ADG/nr3d_lib",
    # license="BSD 3-Clause \"New\" or \"Revised\" License",
    # packages=["nr3d_lib"],
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension}
)
