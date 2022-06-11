#!/opt/homebrew/bin/bash

# python -c "from astronet.tests.conftest import ISA;print(ISA)"
# ISA=`uname -m`
# if [[ -n "$1" ]]; then
#   echo "Generating hash for new plot..."
#   pytest --mpl-generate-path=baseline test_plots.py
#   pytest --mpl-generate-hash-library=astronet/tests/unit/viz/baseline/$ISA-hashlib.json test_plots.py
#   pytest test_plots.py
# else
#   echo "Testing against images in hashlib..."
#   pytest test_plots.py
# fi

# pytest --mpl-generate-path=baseline test_plots.py

# ISA=`uname -m`
# pytest --mpl-generate-hash-library=./baseline/$ISA-hashlib.json test_plots.py

# pytest test_plots.py

pytest \
  --mpl-baseline-path=astronet/tests/reg/baseline/ \
  --mpl-generate-hash-library=astronet/tests/reg/baseline/arm64-hashlib.json \
  --mpl-hash-library=astronet/tests/reg/baseline/arm64-hashlib.json \
  --mpl-results-always \
  test_plots.py
