# Contributing to `astronet`

Firstly, thanks for considering making a contribution!

## Development Setup

astronet uses [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- uv (install instructions below)

### Install uv

On macOS and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/tallamjr/astronet.git
cd astronet

# Install dependencies (creates .venv automatically)
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

## Pull Requests

For code changes, please submit a pull request (PR) by:

1. Making a relevant `issue` (please use the suitable template)
2. Forking this repository
3. Create a new branch labelled as: `issues/<issue-number>/<short-description>`
   ```bash
   git checkout -b issues/123/add-new-feature
   ```
4. Make your changes and commit with conventional commit messages:
   ```bash
   git commit -m "feat: add new feature"
   git commit -m "fix: resolve bug in preprocessing"
   git commit -m "docs: update installation instructions"
   ```
5. Ensure branch is up to date with `master`:
   ```bash
   git fetch origin
   git rebase origin/master
   ```
6. Run tests and quality checks:
   ```bash
   uv run pytest
   uv run black --check .
   uv run ruff check .
   uv run mypy astronet
   ```
7. Push your branch and create a PR on GitHub

## Tests

For any code changes, please run the test suite:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest astronet/tests/unit/test_specific.py

# Run with coverage
uv run pytest --cov=astronet --cov-report=html

# Run only fast tests
uv run pytest -m "not slow"

# Run tests in parallel
uv run pytest -n auto
```

See [`astronet/tests/README.md`](https://github.com/tallamjr/astronet/blob/master/astronet/tests/README.md) for more details.

## Code Quality

### Formatting

Code formatting is enforced using [`black`](https://github.com/psf/black):

```bash
# Check formatting
uv run black --check .

# Apply formatting
uv run black .
```

### Linting

Linting is done using [`ruff`](https://github.com/astral-sh/ruff):

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .
```

### Type Checking

Type hints are checked using [`mypy`](https://mypy.readthedocs.io/):

```bash
uv run mypy astronet
```

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Install hooks
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files
```

## Adding Dependencies

Use uv to manage dependencies:

```bash
# Add runtime dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Add with version constraint
uv add 'package-name>=1.0,<2.0'
```

Dependencies are automatically added to `pyproject.toml` and locked in `uv.lock`.

## Editor Setup

### VSCode

Recommended extensions:
- Python
- Pylance
- Black Formatter
- Ruff

Add to `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### Vim/Neovim

Install the black plugin:
```vim
" Using vim-plug
Plug 'psf/black', { 'branch': 'stable' }

" Format on save
autocmd BufWritePre *.py execute ':Black'

" Manual format shortcuts
xnoremap <Leader>k :!black -q -<CR>
map <Leader>kk :Black<CR>
```

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Example:
```bash
git commit -m "feat: add PyTorch implementation of Tinho model"
git commit -m "fix: correct GP interpolation for sparse time series"
git commit -m "docs: update installation guide for uv"
```

## Further Reading

- [Development Workflow Guide](docs/development/uv-workflow.md)
- [Architecture Decision Records](docs/architecture/decisions/)
- [Testing Guide](astronet/tests/README.md)

## Questions?

Open an issue or discussion on GitHub!
