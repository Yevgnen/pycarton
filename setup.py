# -*- coding: utf-8 -*-

import os
import re

from setuptools import find_packages, setup


def find_version(*paths):
    with open(os.path.join(os.path.dirname(__file__), *paths)) as f:
        text = f.read()
        match = re.search(
            r'^__version__ = (?P<quote>["\'])(?P<ver>[^"\']+)(?P=quote)', text, re.M
        )
        if not match:
            raise RuntimeError("Unable to find version string.")

        return match.group("ver")


VERSION = find_version("carton", "__init__.py")

setup(
    name="carton",
    description="Python toolbox.",
    version=VERSION,
    url="https://github.com/Yevgnen/carton",
    author="Yevgnen Koh",
    author_email="wherejoystarts@gmail.com",
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data={
        "carton": ["py.typed"],
    },
    include_package_data=True,
    install_requires=[
        "pytoml",
        "ruyaml",
    ],
    test_suite="tests",
    zip_safe=False,
)
