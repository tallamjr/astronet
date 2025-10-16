from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astronet")
except PackageNotFoundError:
    __version__ = "0.10.0.dev"
