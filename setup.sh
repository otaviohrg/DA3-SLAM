#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Install Python dependencies (includes torch, required as a build dep by gsplat)
echo "Installing base requirements..."
uv pip install -r requirements.txt

mkdir -p /opt/third_party

# 3. Clone and install Depth Anything 3
echo "Cloning and installing Depth Anything 3..."
cd /opt/third_party
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
uv pip install --no-build-isolation -e "./Depth-Anything-3[all]"
cd -

# 4. Clone and install SALAD
echo "Cloning and installing Salad..."
cd /opt/third_party
git clone https://github.com/Dominic101/salad.git
uv pip install -e ./salad
uv pip install pytorch_lightning pytorch_metric_learning
cd -

# 5. Install da3_slam package
echo "Installing da3_slam..."
uv pip install -e /app
