from pip.req import parse_requirements
from setuptools import setup, find_packages

install_reqs = parse_requirements("requirements.txt")

REQUIREMENTS = [str(ir.req) for ir in install_reqs]

__FALLBACK_VERSION__ = "0.1"

setup(
    name="astronet",
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "fallback_version": __FALLBACK_VERSION__,
    },
    setup_requires=["setuptools_scm>=3.5.0"],
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
)
