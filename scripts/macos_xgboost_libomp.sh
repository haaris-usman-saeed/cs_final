#!/usr/bin/env bash
# macOS only: XGBoost wheels link @rpath/libomp.dylib and expect Homebrew's libomp.
# Instead of installing system libraries, point the venv's libxgboost at the libomp that
# scikit-learn's wheel already bundles, then ad-hoc re-sign. Affects only this virtualenv.
# (Alternative: `brew install libomp`.)
set -euo pipefail
PY=${PYTHON:-.venv/bin/python}
SK=$($PY -c "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))")
XG=$($PY -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('xgboost').origin))")/lib/libxgboost.dylib
install_name_tool -add_rpath "$SK" "$XG" 2>/dev/null || true
codesign --force -s - "$XG"
$PY -c "import xgboost; print('xgboost', xgboost.__version__, 'loads OK')"
