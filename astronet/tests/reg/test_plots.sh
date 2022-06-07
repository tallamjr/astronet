#!/opt/homebrew/bin/bash

# python -c "from astronet.tests.conftest import ISA;print(ISA)"
ISA=`uname -m`
if [[ -n "$1" ]]; then
  echo "Generating hash for new plot..."
  pytest --mpl-generate-path=baseline test_plots.py
  pytest --mpl-generate-hash-library=astronet/tests/unit/viz/baseline/$ISA-hashlib.json test_plots.py
  pytest test_plots.py
else
  echo "Testing against images in hashlib..."
  pytest test_plots.py
fi
