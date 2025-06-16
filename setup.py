from setuptools import setup, Extension, find_packages

# Update this to the path where your yajl headers are located
yajl_include_dir = r"C:\Users\mdogan\Downloads\python-test\yajl\src\yajl"

# List of source files for the yajl2 extension
sources = [
    "src/ijson/backends/ext/_yajl2/yajl_version.c",
    "src/ijson/backends/ext/_yajl2/ijson_yajl2.c",
]

ext_modules = [
    Extension(
        "ijson.backends.ext._yajl2",
        sources=sources,
        include_dirs=[yajl_include_dir],
    )
]

setup(
    name="ijson",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
)
