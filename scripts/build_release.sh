#!/usr/bin/env bash
# Build the compiled release of waveid.crypto
#
# Compiles crypto.py → .so via Cython, then validates by running
# the full test suite against the compiled binary (with .py hidden).
#
# Usage:
#   bash scripts/build_release.sh
#
# Prerequisites:
#   pip install cython numpy scipy

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRYPTO_PY="$PROJECT_ROOT/src/waveid/crypto.py"
CRYPTO_BAK="$PROJECT_ROOT/src/waveid/crypto.py.bak"

echo "=== WaveID Release Build ==="
echo ""

# 1. Check dependencies
echo "[1/5] Checking dependencies..."
python -c "import Cython" 2>/dev/null || { echo "ERROR: Cython not installed. Run: pip install cython"; exit 1; }
python -c "import numpy" 2>/dev/null || { echo "ERROR: NumPy not installed."; exit 1; }
echo "       OK"

# 2. Compile crypto.py -> .so
echo "[2/5] Compiling crypto.py -> .so ..."
cd "$PROJECT_ROOT"
python setup_cython.py build_ext --inplace 2>&1 | tail -3

# 3. Verify .so was produced
echo "[3/5] Verifying build output..."
SO_FILE=$(find src/waveid -maxdepth 1 -name "crypto.cpython-*.so" -type f | head -1)
if [ -z "$SO_FILE" ]; then
    echo "ERROR: .so file not generated"
    exit 1
fi
SO_SIZE=$(du -h "$SO_FILE" | cut -f1)
echo "       Built: $SO_FILE ($SO_SIZE)"

# 4. Test against compiled binary (hide .py source)
echo "[4/5] Running tests against compiled binary..."
mv "$CRYPTO_PY" "$CRYPTO_BAK"

# Clear any cached .pyc so Python doesn't use the old bytecode
find "$PROJECT_ROOT/src/waveid" -name "crypto*.pyc" -delete 2>/dev/null || true
find "$PROJECT_ROOT/src/waveid/__pycache__" -name "crypto*" -delete 2>/dev/null || true

TEST_RESULT=0
python -m pytest tests/ -v || TEST_RESULT=$?

# Restore source
mv "$CRYPTO_BAK" "$CRYPTO_PY"

if [ $TEST_RESULT -ne 0 ]; then
    echo ""
    echo "ERROR: Tests FAILED with compiled .so"
    exit 1
fi

# 5. Summary
echo ""
echo "[5/5] Build complete"
echo "       Binary:   $SO_FILE"
echo "       Tests:    ALL PASSED"
echo ""
echo "To prepare main branch for release:"
echo "  1. git checkout main"
echo "  2. rm src/waveid/crypto.py"
echo "  3. git add src/waveid/crypto.cpython-*.so"
echo "  4. git commit -m 'Release: replace crypto source with compiled binary'"
echo "  5. git push"
