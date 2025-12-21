from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "ec_ls",
        ["ls_module.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    )
]

setup(name="ec_ls", version="0.5", ext_modules=ext_modules)
