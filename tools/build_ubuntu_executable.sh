#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! python3 -c 'import rclpy, std_msgs, torch, numpy' >/dev/null 2>&1; then
  echo "ERROR: source your ROS2 environment first and install Python dependencies." >&2
  echo "Example: source /opt/ros/<distro>/setup.bash" >&2
  exit 1
fi

python3 -m pip install --upgrade pyinstaller

rm -rf build dist
python3 -m PyInstaller   --noconfirm   --clean   --onefile   --name marl2d_node   --paths "$ROOT/src"   --collect-submodules marl2d   --collect-all rclpy   --collect-all std_msgs   --collect-all torch   "$ROOT/tools/marl2d_node_entry.py"

mkdir -p dist/deployment
cp dist/marl2d_node dist/deployment/marl2d_node
cp deploy/reward.py dist/deployment/reward.py
cp deploy/config.json dist/deployment/config.json

echo
echo "Deployment bundle:"
echo "  $ROOT/dist/deployment/marl2d_node"
echo "  $ROOT/dist/deployment/reward.py"
echo "  $ROOT/dist/deployment/config.json"
echo
echo "Copy those three files to each Ubuntu computer."
