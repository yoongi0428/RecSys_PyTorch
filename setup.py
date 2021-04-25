from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

pyx_directories = ["evaluation/backend/cython"]
cpp_dirs = ["evaluation/backend/cython/include"]

pwd = os.getcwd()

additional_dirs = [os.path.join(pwd, d) for d in cpp_dirs]

for t_dir in pyx_directories:
    target_dir = os.path.join(pwd, t_dir)
    # os.chdir(target_dir)
    ori_files = set(os.listdir(t_dir))

    extensions = [
    Extension(
        '*',
        [os.path.join(t_dir, "*.pyx")],
        extra_compile_args=["-std=c++11"])
    ]
    setup(
        ext_modules=cythonize(extensions, language="c++"),
        include_dirs=[np.get_include()]+additional_dirs
    )

    new_files = set(os.listdir(t_dir))
    for n_file in new_files:
        if n_file not in ori_files and n_file.split(".")[-1] in ("c", "cpp"):
            os.remove(os.path.join(t_dir, n_file))

    # os.chdir(pwd)