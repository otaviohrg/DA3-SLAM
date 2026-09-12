#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Install Python dependencies (includes torch, required as a build dep by gsplat)
echo "Installing base requirements..."
uv pip install -r requirements.txt

# 1b. Optional live-demo requirements (RealSense + Rerun). Present in the Docker
# image; skipped gracefully if the file was not copied in (e.g. local runs).
if [ -f requirements-realsense.txt ]; then
    echo "Installing RealSense demo requirements..."
    uv pip install -r requirements-realsense.txt
fi

mkdir -p /opt/third_party

# Pinned upstream commits.  Both repos are cloned from git rather than PyPI, so
# an unpinned `main` makes the image non-reproducible: a rebuild can silently
# change the DA3 backbone (moving every benchmark number) or the SALAD
# descriptor.  These are the commits the validated results were produced with.
# The token-merging port reaches into DA3's ViT internals and guards itself with
# _verify_upstream() precisely because this checkout is re-cloned — bumping the
# pin means re-running that check.
# Override to track upstream, e.g. DA3_COMMIT=main ./setup.sh
DA3_COMMIT="${DA3_COMMIT:-41736238f5bced4debf3f2a12375d2466874866d}"
SALAD_COMMIT="${SALAD_COMMIT:-33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d}"

# 3. Clone and install Depth Anything 3
echo "Cloning and installing Depth Anything 3 (pin: $DA3_COMMIT)..."
cd /opt/third_party
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
git -C Depth-Anything-3 checkout --quiet "$DA3_COMMIT"
uv pip install --no-build-isolation -e "./Depth-Anything-3[all]"
cd -

# 4. Clone and install SALAD
echo "Cloning and installing Salad (pin: $SALAD_COMMIT)..."
cd /opt/third_party
git clone https://github.com/Dominic101/salad.git
git -C salad checkout --quiet "$SALAD_COMMIT"
uv pip install -e ./salad
uv pip install pytorch_lightning pytorch_metric_learning
cd -

# 5. Install da3_slam package
echo "Installing da3_slam..."
uv pip install -e /app
