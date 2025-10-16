# Development Workflow with uv

This document describes the development workflow for astronet using `uv`, the fast Python package manager.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Development Workflow](#development-workflow)
- [Running Tests](#running-tests)
- [Code Quality](#code-quality)
- [Managing Dependencies](#managing-dependencies)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- uv (installation instructions below)

## Installation

### Install uv

On macOS and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or using pip:
```bash
pip install uv
```

### Clone and Set Up the Project

```bash
# Clone the repository
git clone https://github.com/tallamjr/astronet.git
cd astronet

# Create a virtual environment and install dependencies
uv sync

# This creates a .venv directory and installs all dependencies from uv.lock
```

## Development Workflow

### Activate the Virtual Environment

uv automatically manages the virtual environment, but if you need to activate it manually:

```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### Run Commands in the Environment

You can use `uv run` to execute commands without activating the environment:

```bash
# Run Python scripts
uv run python script.py

# Run pytest
uv run pytest

# Run any installed command-line tool
uv run black .
```

## Running Tests

### Run All Tests

```bash
uv run pytest
```

### Run Specific Test Files

```bash
uv run pytest astronet/tests/unit/test_specific.py
```

### Run Tests with Coverage

```bash
uv run pytest --cov=astronet --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Tests in Parallel

```bash
uv run pytest -n auto
```

### Run Only Fast Tests

```bash
uv run pytest -m "not slow"
```

## Code Quality

### Format Code with Black

```bash
# Check formatting
uv run black --check .

# Apply formatting
uv run black .
```

### Lint with Ruff

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .
```

### Type Checking with mypy

```bash
uv run mypy astronet
```

### Run All Quality Checks

```bash
uv run black --check .
uv run ruff check .
uv run mypy astronet
uv run pytest
```

### Pre-commit Hooks

Install pre-commit hooks to run checks automatically:

```bash
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

## Managing Dependencies

### Add a New Dependency

```bash
# Add to main dependencies
uv add package-name

# Add to dev dependencies
uv add --dev package-name

# Add with version constraints
uv add 'package-name>=1.0,<2.0'
```

### Remove a Dependency

```bash
uv remove package-name
```

### Update Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package package-name

# Sync environment with updated lock file
uv sync
```

### Install Optional Dependency Groups

```bash
# Install TensorFlow support (for legacy code)
uv sync --extra tensorflow

# Install PyTorch support (for v2.0)
uv sync --extra pytorch

# Install data processing tools
uv sync --extra data

# Install all extras
uv sync --all-extras
```

## Common Tasks

### Run Jupyter Notebooks

```bash
# Install jupyter extras if not already installed
uv sync --extra jupyter

# Start JupyterLab
uv run jupyter lab
```

### Build Documentation

```bash
# Install docs extras if not already installed
uv sync --extra docs

# Build docs
uv run mkdocs build

# Serve docs locally
uv run mkdocs serve
```

### Profile Code

```bash
# Install profiling extras if not already installed
uv sync --extra profiling

# Profile with py-spy
uv run py-spy record -o profile.svg -- python script.py

# Profile memory with memray
uv run memray run script.py
uv run memray flamegraph memray-output.bin
```

### Training Models

```bash
# Run training script (example)
uv run python astronet/scripts/train.py \
  --config configs/models/tinho.yaml \
  --dataset plasticc
```

### Export Models

```bash
# Export to ONNX (example)
uv run python astronet/scripts/export.py \
  --model-path models/tinho-best.ckpt \
  --output-path models/tinho.onnx \
  --format onnx
```

## Troubleshooting

### Lock File Out of Sync

If you see errors about the lock file being out of sync:

```bash
uv sync --refresh
```

### Platform-Specific Issues

If you encounter platform-specific dependency issues:

```bash
# Force reinstall
uv sync --reinstall

# Clear cache
uv cache clean
```

### Python Version Issues

Check your Python version:

```bash
python --version
```

astronet requires Python 3.10, 3.11, or 3.12. To use a specific version:

```bash
uv venv --python 3.11
uv sync
```

### Virtual Environment Issues

If the virtual environment is corrupted:

```bash
# Remove the virtual environment
rm -rf .venv

# Recreate it
uv sync
```

## Performance Tips

### Use Locked Dependencies

Always use `--frozen` in CI/CD to ensure reproducibility:

```bash
uv sync --frozen
```

### Cache in CI/CD

The GitHub Actions workflow caches uv installations:

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v5
  with:
    enable-cache: true
    cache-dependency-glob: "uv.lock"
```

### Parallel Testing

Use pytest-xdist for faster test runs:

```bash
uv run pytest -n auto
```

## Comparison with pip/conda

| Task | pip/conda | uv |
|------|-----------|-----|
| Install deps | `pip install -r requirements.txt` | `uv sync` |
| Add dependency | Edit requirements.txt + `pip install` | `uv add package` |
| Run command | `source venv/bin/activate && python` | `uv run python` |
| Lock dependencies | Manual with pip-tools | Automatic with uv.lock |
| Speed | Slow (minutes) | Fast (seconds) |
| Reproducibility | Manual | Automatic |

## Why uv?

- **10-100x faster** than pip for dependency resolution
- **Built-in lockfile** support for reproducible installs
- **Zero-dependency** installation (single binary)
- **Drop-in replacement** for pip, pip-tools, and virtualenv
- **Better caching** reduces redundant downloads
- **Modern pyproject.toml** support

## Further Reading

- [uv Documentation](https://docs.astral.sh/uv/)
- [pyproject.toml specification](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/)
- [astronet Contributing Guide](../../CONTRIBUTING.md)
