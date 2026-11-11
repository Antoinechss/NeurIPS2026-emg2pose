#!/usr/bin/env bash
# Install everything needed to train and evaluate:
#   - the organisers' benchmark, pinned to the commit we used, in third_party/
#   - a Python 3.12 virtualenv (.venv) with the pinned packages
#   - env.sh, to source in every new shell
# Usage: bash scripts/setup.sh [data_root]
#   data_root holds the downloaded data and caches (default: ./data).
#   On RunPod, pass the network volume: bash scripts/setup.sh /workspace
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
mkdir -p "${1:-data}"
DATA_ROOT=$(cd "${1:-data}" && pwd)

BENCH_URL=https://github.com/x-datascience-datacamp/codabench_emg2pose.git
BENCH_COMMIT=6cd689804686b71e432b706ea1ab5f04ba2350b2
BENCH_DIR=third_party/codabench_emg2pose
[ -d "$BENCH_DIR" ] || git clone -q "$BENCH_URL" "$BENCH_DIR"
git -C "$BENCH_DIR" checkout -q "$BENCH_COMMIT"

# uv downloads Python 3.12 (neuralset needs >= 3.12). Its Python and cache go
# under data_root, so they survive pod restarts when data_root is a volume.
export UV_PYTHON_INSTALL_DIR="$DATA_ROOT/uv-python" UV_CACHE_DIR="$DATA_ROOT/.uv-cache"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$DATA_ROOT/bin" INSTALLER_NO_MODIFY_PATH=1 sh
    export PATH="$DATA_ROOT/bin:$PATH"
fi
[ -d .venv ] || uv venv -q -p 3.12 .venv
VIRTUAL_ENV="$ROOT/.venv" uv pip install -q -r requirements.txt

# neuralbench reads its folders from a settings file (else it prompts): the
# organisers' file, with its folders moved under data_root.
python3 - "$BENCH_DIR/tools/neuralbench_config.json" "$DATA_ROOT" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
for key in ("CACHE_DIR", "SAVE_DIR", "DATA_DIR"):
    cfg[key] = cfg[key].replace("/tmp/neuralbench", sys.argv[2] + "/neuralbench")
json.dump(cfg, open("neuralbench_config.json", "w"), indent=2)
PY

cat > env.sh <<EOF
export BENCHOPT_DATA_HOME=$DATA_ROOT/data
export NEURALBENCH_CONFIG=$ROOT/neuralbench_config.json
# Same as the organisers' Docker image: parallel extraction corrupts the data
# index on many-core machines.
export NEURALBENCH_EXTRACT_CPUS=1
export UV_PYTHON_INSTALL_DIR=$DATA_ROOT/uv-python UV_CACHE_DIR=$DATA_ROOT/.uv-cache
export PATH="$DATA_ROOT/bin:\$PATH"
export B=$ROOT/$BENCH_DIR/benchmark
. $ROOT/.venv/bin/activate
EOF
. ./env.sh
python -c "import torch, braindecode, benchopt; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo "Setup OK. In every new shell: source env.sh   (the benchmark folder is \$B)"
