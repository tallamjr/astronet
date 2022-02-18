from setuptools import find_packages, setup

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

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
