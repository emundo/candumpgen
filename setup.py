# pylint: disable=exec-used
import os
from typing import Dict, List, Union

from setuptools import setup, find_packages

source_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candumpgen")

version_scope: Dict[str, Dict[str, str]] = {}
with open(os.path.join(source_root, "version.py")) as f:
    exec(f.read(), version_scope)
version = version_scope["__version__"]

project_scope: Dict[str, Dict[str, Union[str, List[str]]]] = {}
with open(os.path.join(source_root, "project.py")) as f:
    exec(f.read(), project_scope)
project = project_scope["project"]

with open("README.md") as f:
    long_description = f.read()

classifiers = [
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",

    "License :: OSI Approved :: Apache Software License",

    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",

    "Typing :: Typed"
]

classifiers.extend(project["categories"])

if version["tag"] == "alpha":
    classifiers.append("Development Status :: 3 - Alpha")

if version["tag"] == "beta":
    classifiers.append("Development Status :: 4 - Beta")

if version["tag"] == "stable":
    classifiers.append("Development Status :: 5 - Production/Stable")

del project["categories"]
del project["year"]

setup(
    version=version["short"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "candumpgen=candumpgen.__main__:main"
        ],
    },
    install_requires=[
        "cantools>=35.3.0,<36",
        "python-can>=3.3.4,<4",
        "scipy>=1.5.3,<2"
    ],
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
    classifiers=classifiers,
    **project
)
