from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

__FALLBACK_VERSION__ = "0.1"

setup(
    name="astronet",
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "fallback_version": __FALLBACK_VERSION__},
    setup_requires=['setuptools_scm>=3.5.0'],
    packages=find_packages(),
    install_requires=requirements
)
