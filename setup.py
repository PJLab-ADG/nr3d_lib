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
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CSRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, 'csrc'))
EXTERNALS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, 'externals'))
"""
NOTE:
`sources` expects relative directories (preferrable to achieve the most compatibility);
`include_dirs` expects absolute directories (must be)
"""

# Read version information
with open(os.path.join(SCRIPT_DIR, 'VERSION'),"r") as f: 
    VERSION = f.read()

COMMON_LIB_DIRS = []
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

    # Custom options read from env
    if os.environ.get('FIX_LCUDA', False):
        # NOTE: On cluster machines's login node etc. where no GPU is installed, 
        #           `libcuda.so` will not be in usally place, 
        #           so `-lcuda` will raise "can not find -lcuda" error.
        #       To solve this, we can use `libcuda.so` in the `stubs` directory.
        # https://stackoverflow.com/questions/62999715/when-i-make-darknet-with-cuda-1-usr-bin-ld-cannot-find-lcudaoccured-how
        COMMON_LIB_DIRS.append(os.path.join(os.environ.get('CUDA_HOME'), 'lib64', 'stubs'))

    if os.environ.get('USE_CPP17', False):
        cpp_standard = "c++17"
    else:
        cpp_standard = "c++14"

    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor

    print(f"=> Targeting compute capability {compute_capability}")

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
        f"-std={cpp_standard}",
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
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]

    definitions = [
        "-O3",
        "-DNDEBUG"
    ]
    nvcc_flags += definitions
    c_flags += definitions

    # List of sources.
    sources = [
        os.path.join("csrc", "lotd", "src", "compile_split_1.cu"),
        os.path.join("csrc", "lotd", "src", "compile_split_2.cu"),
        os.path.join("csrc", "lotd", "src", "compile_split_3.cu"),
        os.path.join("csrc", "lotd", "src", "lotd_torch_api.cu"),
        os.path.join("csrc", "lotd", "src", "lotd.cpp"),
    ]
    include_dirs = [
        os.path.join(CSRC_DIR, "lotd", "include"), 
        os.path.join(CSRC_DIR, "forest")
    ]
    libraries = []
    library_dirs = deepcopy(COMMON_LIB_DIRS)
    
    extra_objects = []
    ext = CUDAExtension(
        name="nr3d_lib.bindings._lotd",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
        libraries=libraries,
        library_dirs=library_dirs,
        extra_objects=extra_objects
    )
    return ext

def get_ext_pack_ops():
    nvcc_flags = [
        '-O3', f"-std={cpp_standard}",
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
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]

    if '--use-thrust-sort' in sys.argv:
        sys.argv.remove('--use-thrust-sort')
        c_flags.append("-DPACK_OPS_USE_THRUST_SORT")
        nvcc_flags.append("-DPACK_OPS_USE_THRUST_SORT")

    # List of sources.
    sources = [
        os.path.join("csrc", 'pack_ops', 'pack_ops_cuda.cu'), 
        os.path.join("csrc", 'pack_ops', 'pack_ops.cpp'), 
    ]
    include_dirs = [
        os.path.join(CSRC_DIR, 'pack_ops')
    ]
    library_dirs = deepcopy(COMMON_LIB_DIRS)

    ext = CUDAExtension(
        name='nr3d_lib.bindings._pack_ops', # extension name, import this to use CUDA API
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags,}, 
        library_dirs=library_dirs
    )
    return ext

def get_ext_forest():
    nvcc_flags = [
        '-O3', f"-std={cpp_standard}",
        # NOTE: We do not want built-in half ops. use cuda's half ops instead.
        #       https://discuss.pytorch.org/t/error-more-than-one-operator-matches-these-operands-in-thcnumerics-cuh/89935/2
        '-D__CUDA_NO_HALF_OPERATORS__', 
        '-D__CUDA_NO_HALF_CONVERSIONS__', 
        '-D__CUDA_NO_HALF2_OPERATORS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        # NOTE: For using kaolin headers
        '-DWITH_CUDA', 
        '-DTHRUST_IGNORE_CUB_VERSION_CHECK', 
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    
    if os.name == "posix":
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]

    # List of sources.
    sources = [
        os.path.join("csrc", 'forest', 'forest.cpp'),
        os.path.join("externals", 'kaolin_spc_raytrace_fixed', 'src', 'raytrace_cuda.cu'), 
        os.path.join("externals", 'kaolin_spc_raytrace_fixed', 'src', 'raytrace.cpp'), 
    ]
    include_dirs = [
        os.path.join(EXTERNALS_DIR, 'kaolin_spc_raytrace_fixed', 'include'), 
    ]
    if "CUB_HOME" in os.environ:
        logging.warning(f'Including CUB_HOME ({os.environ["CUB_HOME"]}).')
        include_dirs.append(os.environ["CUB_HOME"])
    else:
        _, bare_metal_major, _ = get_cuda_bare_metal_version()
        if int(bare_metal_major) < 11:
            logging.warning(f'Including default CUB_HOME ({os.path.join(EXTERNALS_DIR, "cub")}).')
            include_dirs.append(os.path.join(EXTERNALS_DIR, 'cub'))
    library_dirs = deepcopy(COMMON_LIB_DIRS)

    ext = CUDAExtension(
        name='nr3d_lib.bindings._forest', # extension name, import this to use CUDA API
        sources=sources,
        include_dirs=include_dirs, 
        extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags,}, 
        define_macros=[("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)], 
        library_dirs=library_dirs
    )
    return ext

def get_ext_permuto():
    nvcc_flags = [
        f"-std={cpp_standard}",
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
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]

    definitions = [
    ]
    nvcc_flags += definitions
    c_flags += definitions

    # List of sources.
    sources = [
        os.path.join("csrc", "permuto", "src", "compile_split_1.cu"),
        os.path.join("csrc", "permuto", "src", "compile_split_2.cu"),
        os.path.join("csrc", "permuto", "src", "compile_split_3.cu"),
        os.path.join("csrc", "permuto", "src", "compile_split_4.cu"),
        os.path.join("csrc", "permuto", "src", "compile_split_5.cu"),
        os.path.join("csrc", "permuto", "src", "compile_split_6.cu"),
        os.path.join("csrc", "permuto", "src", "permuto_cuda.cu"),
        os.path.join("csrc", "permuto", "src", "permuto.cpp"),
    ]
    include_dirs = [
        os.path.join(CSRC_DIR, "permuto", "include"), 
    ]
    libraries = []
    library_dirs = deepcopy(COMMON_LIB_DIRS)
    extra_objects = []

    ext = CUDAExtension(
        name="nr3d_lib.bindings._permuto",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
        libraries=libraries,
        library_dirs=library_dirs,
        extra_objects=extra_objects
    )
    return ext

def get_ext_permuto_quicksort():
    nvcc_flags = [
        f"-std={cpp_standard}",
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
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]

    definitions = [
    ]
    nvcc_flags += definitions
    c_flags += definitions

    # List of sources.
    sources = [
        os.path.join("csrc", "permuto_quicksort", "src", "compile_split_1.cu"),
        os.path.join("csrc", "permuto_quicksort", "src", "compile_split_2.cu"),
        os.path.join("csrc", "permuto_quicksort", "src", "compile_split_3.cu"),
        os.path.join("csrc", "permuto_quicksort", "src", "compile_split_4.cu"),
        os.path.join("csrc", "permuto_quicksort", "src", "compile_split_5.cu"),
        os.path.join("csrc", "permuto_quicksort", "src", "compile_split_6.cu"),
        os.path.join("csrc", "permuto_quicksort", "src", "permuto_cuda.cu"),
        os.path.join("csrc", "permuto_quicksort", "src", "permuto.cpp"),
    ]
    include_dirs = [
        os.path.join(CSRC_DIR, "permuto_quicksort", "include"), 
    ]
    libraries = []
    library_dirs = deepcopy(COMMON_LIB_DIRS)
    extra_objects = []

    ext = CUDAExtension(
        name="nr3d_lib.bindings._permuto_quicksort",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
        libraries=libraries,
        library_dirs=library_dirs,
        extra_objects=extra_objects
    )
    return ext

def get_ext_permuto_thrust():
    nvcc_flags = [
        f"-std={cpp_standard}",
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
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]

    definitions = [
    ]
    nvcc_flags += definitions
    c_flags += definitions

    # List of sources.
    sources = [
        os.path.join("csrc", "permuto_thrust", "src", "compile_split_1.cu"),
        os.path.join("csrc", "permuto_thrust", "src", "compile_split_2.cu"),
        os.path.join("csrc", "permuto_thrust", "src", "compile_split_3.cu"),
        os.path.join("csrc", "permuto_thrust", "src", "compile_split_4.cu"),
        os.path.join("csrc", "permuto_thrust", "src", "compile_split_5.cu"),
        os.path.join("csrc", "permuto_thrust", "src", "compile_split_6.cu"),
        os.path.join("csrc", "permuto_thrust", "src", "permuto_cuda.cu"),
        os.path.join("csrc", "permuto_thrust", "src", "permuto.cpp"),
    ]
    include_dirs = [
        os.path.join(CSRC_DIR, "permuto_thrust", "include"), 
    ]
    libraries = []
    library_dirs = deepcopy(COMMON_LIB_DIRS)
    extra_objects = []

    ext = CUDAExtension(
        name="nr3d_lib.bindings._permuto_thrust",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
        libraries=libraries,
        library_dirs=library_dirs,
        extra_objects=extra_objects
    )
    return ext

def get_ext_permuto_intermediate():
    nvcc_flags = [
        f"-std={cpp_standard}",
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
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]

    definitions = [
    ]
    nvcc_flags += definitions
    c_flags += definitions

    # List of sources.
    sources = [
        os.path.join("csrc", "permuto_intermediate", "src", "permuto_cuda.cu"),
        os.path.join("csrc", "permuto_intermediate", "src", "permuto.cpp"),
    ]
    include_dirs = [
        os.path.join(CSRC_DIR, "permuto_intermediate", "include"), 
    ]
    libraries = []
    library_dirs = deepcopy(COMMON_LIB_DIRS)
    extra_objects = []

    ext = CUDAExtension(
        name="nr3d_lib.bindings._permuto_intermediate",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
        libraries=libraries,
        library_dirs=library_dirs,
        extra_objects=extra_objects
    )
    return ext

def get_ext_occ_grid():
    nvcc_flags = [
        '-O3', f"-std={cpp_standard}",
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
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]

    # List of sources.
    sources = [
        os.path.join("csrc", 'occ_grid', 'src', 'ray_marching.cu'), 
        os.path.join("csrc", 'occ_grid', 'src', 'batched_marching.cu'), 
        os.path.join("csrc", 'occ_grid', 'src', 'forest_marching.cu'), 
        os.path.join("csrc", 'occ_grid', 'src', 'occ_grid.cpp'), 
    ]
    include_dirs = [
        os.path.join(CSRC_DIR, "occ_grid", "include"), 
        os.path.join(CSRC_DIR, "forest") # For forest occ grid marching
    ]
    library_dirs = deepcopy(COMMON_LIB_DIRS)

    ext = CUDAExtension(
        name='nr3d_lib.bindings._occ_grid', # extension name, import this to use CUDA API
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags,}, 
        library_dirs=library_dirs
    )
    return ext

def get_ext_spherical_embedder():
    # Modified from https://github.com/ashawkey/torch-ngp
    nvcc_flags = [
        '-O3', f"-std={cpp_standard}",
        '-U__CUDA_NO_HALF_OPERATORS__', 
        '-U__CUDA_NO_HALF_CONVERSIONS__', 
        '-U__CUDA_NO_HALF2_OPERATORS__',
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    
    if os.name == "posix":
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]
    
    # List of sources.
    sources = [
        os.path.join("externals", 'shencoder', 'shencoder.cu'), 
        os.path.join("externals", 'shencoder', 'bindings.cpp'), 
    ]
    library_dirs = deepcopy(COMMON_LIB_DIRS)
    
    ext = CUDAExtension(
        name='nr3d_lib.bindings._shencoder', # extension name, import this to use CUDA API
        sources=sources,
        extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags,}, 
        library_dirs=library_dirs
    )
    return ext

def get_ext_frequency_embedder():
    # Modified from https://github.com/ashawkey/torch-ngp
    nvcc_flags = [
        '-O3', f"-std={cpp_standard}",
        '-U__CUDA_NO_HALF_OPERATORS__', 
        '-U__CUDA_NO_HALF_CONVERSIONS__', 
        '-U__CUDA_NO_HALF2_OPERATORS__',
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    
    if os.name == "posix":
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]
    
    # List of sources.
    sources = [
        os.path.join("externals", 'freqencoder', 'freqencoder.cu'), 
        os.path.join("externals", 'freqencoder', 'bindings.cpp')
    ]
    library_dirs = deepcopy(COMMON_LIB_DIRS)
    
    ext = CUDAExtension(
        name='nr3d_lib.bindings._freqencoder', # extension name, import this to use CUDA API
        sources=sources,
        extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags,}, 
        library_dirs=library_dirs
    )
    return ext

def get_ext_pytorch3d_knn():
    nvcc_flags = [
        f"-std={cpp_standard}",
        
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]

    if os.name == "posix":
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            '-Xcompiler="-mf16c"',
            '-Xcompiler="-Wno-float-conversion"',
            '-Xcompiler="-fno-strict-aliasing"',
            '-Xcompiler="-Wno-unused-parameter"', 
            "-Xcudafe=--diag_suppress=declared_but_not_referenced", 
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]

    # List of sources.
    sources = [
        os.path.join("externals", 'pytorch3d_knn', 'knn.cu'), 
        os.path.join("externals", 'pytorch3d_knn', 'knn_cpu.cpp'), 
        os.path.join("externals", 'pytorch3d_knn', 'ext.cpp')
    ]
    include_dirs = [
        os.path.join(EXTERNALS_DIR, "pytorch3d_knn"), 
    ]
    library_dirs = deepcopy(COMMON_LIB_DIRS)
    ext = CUDAExtension(
        name='nr3d_lib.bindings._pytorch3d_knn', # extension name, import this to use CUDA API
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags,}, 
        define_macros=[("WITH_CUDA", None)], 
        library_dirs=library_dirs
    )
    return ext

def get_ext_spheretrace():
    nvcc_flags = [
        f"-std={cpp_standard}",
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
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
            # "-Xcudafe=--diag_suppress=20012",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]

    definitions = [
        f"-DTCNN_MIN_GPU_ARCH={compute_capability}",
        "-O3",
        "-DNDEBUG"
    ]
    nvcc_flags += definitions
    c_flags += definitions

    # List of sources.
    sources = [
        os.path.join("csrc", "sphere_trace", "src", "entry.cu"),
        os.path.join("csrc", "sphere_trace", "src", "sphere_tracer.cu"),
        os.path.join("csrc", "sphere_trace", "src", "ray_march.cu"),
    ]
    include_dirs = [
        os.path.join(CSRC_DIR, "sphere_trace", "include"),
        os.path.join(EXTERNALS_DIR, "glm"),
    ]
    libraries = ["cuda"]
    library_dirs = deepcopy(COMMON_LIB_DIRS)
    extra_objects = []

    ext = CUDAExtension(
        name="nr3d_lib.bindings._sphere_trace",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
        libraries=libraries,
        library_dirs=library_dirs,
        extra_objects=extra_objects
    )
    return ext

def get_ext_simple_knn():
    nvcc_flags = [
        f"-std={cpp_standard}",
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    if os.name == "posix":
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
            # "-Xcudafe=--diag_suppress=20012",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}", "/wd4624"]
    
    definitions = [
        "-O3",
    ]
    nvcc_flags += definitions
    c_flags += definitions

    # List of sources.
    sources = [
        os.path.join("externals", "simple_knn", "spatial.cu"), 
        os.path.join("externals", "simple_knn", "simple_knn.cu"), 
        os.path.join("externals", "simple_knn", "ext.cpp"), 
    ]
    ext = CUDAExtension(
        name="nr3d_lib.bindings._simple_knn",
        sources=sources,
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
    )
    return ext


def get_ext_r3dg_rasterization():
    """
    Modified from https://github.com/NJU-3DV/Relightable3DGaussian/blob/main/r3dg-rasterization/setup.py
    """
    nvcc_flags = [
        f"-std={cpp_standard}",
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    if os.name == "posix":
        c_flags = [f"-std={cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
            # "-Xcudafe=--diag_suppress=20012",
        ]
    elif os.name == "nt":
        c_flags = [f"/std:{cpp_standard}"]
    
    definitions = [
        "-O3",
    ]
    nvcc_flags += definitions
    c_flags += definitions
    
    # List of sources.
    sources = [
        os.path.join("externals", "r3dg_rasterization", "cuda_rasterizer", "rasterizer_impl.cu"), 
        os.path.join("externals", "r3dg_rasterization", "cuda_rasterizer", "forward.cu"), 
        os.path.join("externals", "r3dg_rasterization", "cuda_rasterizer", "backward.cu"), 
        os.path.join("externals", "r3dg_rasterization", "rasterize_points.cu"), 
        os.path.join("externals", "r3dg_rasterization", "render_equation.cu"), 
        os.path.join("externals", "r3dg_rasterization", "ext.cpp"), 
    ]
    include_dirs = [
        os.path.join(EXTERNALS_DIR, "glm")
    ]
    
    ext = CUDAExtension(
        name="nr3d_lib.bindings._r3dg_rasterization",
        sources=sources,
        include_dirs=include_dirs, 
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
    )
    return ext

def get_extensions():
    ext_modules = []
    
    #---- Self
    ext_modules.append(get_ext_permuto())
    # ext_modules.append(get_ext_permuto_quicksort())
    # ext_modules.append(get_ext_permuto_thrust())
    # ext_modules.append(get_ext_permuto_intermediate())
    ext_modules.append(get_ext_pack_ops())
    ext_modules.append(get_ext_forest())
    ext_modules.append(get_ext_occ_grid())
    ext_modules.append(get_ext_spheretrace())
    ext_modules.append(get_ext_lotd())

    #---- Externals
    ext_modules.append(get_ext_spherical_embedder())
    ext_modules.append(get_ext_frequency_embedder())
    ext_modules.append(get_ext_simple_knn())
    ext_modules.append(get_ext_pytorch3d_knn())
    ext_modules.append(get_ext_r3dg_rasterization())
    return ext_modules

setup(
    name="nr3d_lib",
    version=VERSION,
    description="nr3d_lib",
    long_description="nr3d_lib",
    keywords="nr3d_lib",
    url="https://github.com/PJLAB-ADG/nr3d_lib",
    author="Jianfei Guo, Nianchen Deng, Xinyang Li, Qiusheng Huang",
    author_email="ffventus@gmail.com, dengnianchen@pjlab.org.cn, lixinyang@pjlab.org.cn, huangqiusheng@pjlab.org.cn",
    maintainer="Jianfei Guo",
    maintainer_email="ffventus@gmail.com",
    download_url=f"https://github.com/PJLAB-ADG/nr3d_lib",
    # license="BSD 3-Clause \"New\" or \"Revised\" License",
    packages=find_packages(),
    install_requires=["ninja"],
    include_package_data=True,
    zip_safe=False,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension}
)
