from setuptools import find_packages, setup

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name="astronet",
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
