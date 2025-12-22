from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "ec_regret",
        ["regret_module.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/O2"],  # MSVC
    )
]

setup(
    name="ec_regret",
    version="0.1",
    ext_modules=ext_modules,
)
