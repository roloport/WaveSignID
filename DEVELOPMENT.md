# WaveSignID Development Guide

## Branch Strategy & IP Protection

WaveSignID uses a **two-branch model** to protect proprietary cryptographic
algorithms while keeping the public repository fully functional.

### Branch Overview

| Branch | Purpose | `crypto.py` | `crypto.*.so` |
|--------|---------|:-----------:|:-------------:|
| `claude/agent-identification-system-*` (dev) | Active development | Source (present) | Not tracked (`.gitignore`) |
| `main` | Public release | **Deleted** | Compiled binary |

### How It Works

- **Dev branch** contains the full Python source of `src/waveid/crypto.py`.
  Only project maintainers with access to this branch can read or modify the
  core cryptographic algorithms.

- **Main branch** ships a compiled native extension (`crypto.cpython-*.so`)
  in place of the Python source. Users who clone `main` can install, run,
  and test the full project — Python imports the `.so` transparently — but
  they cannot inspect the algorithm source code.

### Building a Release

Prerequisites:

```bash
pip install cython numpy scipy
```

Build and validate:

```bash
bash scripts/build_release.sh
```

This script:
1. Compiles `crypto.py` → `.so` via Cython
2. Temporarily hides the `.py` source
3. Runs the full test suite (45 tests) against the compiled binary
4. Restores the `.py` source
5. Reports pass/fail

### Merging to Main (Release Checklist)

After a successful build:

```bash
# 1. Merge dev into main
git checkout main
git merge claude/agent-identification-system-<branch>

# 2. Swap source for compiled binary
rm src/waveid/crypto.py
git add src/waveid/crypto.cpython-*.so
git commit -m "Release: replace crypto source with compiled binary"

# 3. Push
git push origin main
```

### Important Notes

- **Never commit `crypto.py` to `main`.** The source must stay on the dev
  branch only.
- **Never commit the `.so` to the dev branch.** It is listed in `.gitignore`
  so build artifacts are excluded automatically.
- The `.so` is platform-specific (built for the current OS/arch/Python
  version). If you need to support multiple platforms, run `build_release.sh`
  on each target and commit all resulting `.so` files to `main`.
- The compiled binary is ~496KB. Reversing it requires C disassembly —
  significantly harder than reading Python source.

### File Reference

| File | Purpose |
|------|---------|
| `setup_cython.py` | Cython build configuration |
| `scripts/build_release.sh` | Automated build + test script |
| `.gitattributes` | Marks `.so` files as binary in git |
| `.gitignore` | Excludes `.so`, `.c`, `.html` build artifacts on dev |
