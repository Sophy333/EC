from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "ec_amoms",
        ["ec_amoms.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/O2"],  # MSVC optimization
    )
]

setup(
    name="ec_amoms",
    version="0.1",
    ext_modules=ext_modules,
)
