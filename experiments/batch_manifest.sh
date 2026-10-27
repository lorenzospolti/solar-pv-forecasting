#!/usr/bin/env bash
# Usage: bash experiments/batch_manifest.sh <logs-dir> <tag> <env-name> <lock-file> <config> \
#            [command...]
# Writes <logs-dir>/batch_manifest_<tag>.txt (tag: before | after | ...): the environment, the
# git state of this repository, package versions, checkpoint revisions and weight sha256,
# swap (DEFINITIVE_RUN.md section 6). Read-only apart from that one file. Based on the
# 12 Oct manifest of the covariate runs.
set -u
if [ $# -lt 5 ]; then
  sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
LOGS=$1; TAG=$2; ENVNAME=$3; LOCK=$4; CONFIG=$5; shift 5
REPO=$(cd "$(dirname "$0")/.." && pwd)
HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
CONDA=${CONDA_EXE:-$(command -v conda || echo "$HOME/miniforge3/bin/conda")}
PREFIX=$("$CONDA" env list 2>/dev/null | awk -v n="$ENVNAME" '$1 == n {print $NF}')
PY=${PREFIX:+$PREFIX/bin/python}
PY=${PY:-$(command -v python)}
#: the two checkpoints pinned by commit hash, and their weight files' sha256 (HF API, 14 Oct)
TIMESFM3=models--google--timesfm-3.0-pytorch/snapshots/43046b85ec22d584a13f8098c2ed39c889e129c2
TIREX2=models--NX-AI--TiRex-2/snapshots/30788a311699acba7e6393ad9700ea7b7cb9bb2d
SHA_TIMESFM3=a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da
SHA_TIREX2=184b160ffbe4c01a26beeba14015ff3507c7497e1f3577114187bbc1d19fcac1
#: the Chronos-2 pins: darts loads refs/main, so it must name them (fetch_checkpoints.py)
CHRONOS2=29ec3766d36d6f73f0696f85560a422f50e8498c
CHRONOS2_SMALL=ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a
mkdir -p "$LOGS"
OUT="$LOGS/batch_manifest_$TAG.txt"

weight() {  # weight <snapshot-relative file> <expected sha256>
  local f="$HUB/$1" got
  if [ -e "$f" ]; then
    got=$(shasum -a 256 -- "$f" | cut -d' ' -f1)
    echo "  $1: $got $([ "$got" = "$2" ] && echo ok || echo "MISMATCH (expected $2)")"
  else
    echo "  $1: MISSING"
  fi
}

pin() {  # pin <repo cache dir> <pinned commit>: refs/main against the pin
  local got
  got=$(cat "$HUB/$1/refs/main" 2>/dev/null || echo none)
  echo "  $1 refs/main: $got $([ "$got" = "$2" ] && echo ok || echo "MISMATCH (pinned $2)")"
}

{
  echo "written: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "tag: $TAG"
  echo "command: ${*:-(none given)}"
  echo "python: $PY ($("$PY" --version 2>&1)); env $ENVNAME at ${PREFIX:-NOT FOUND}"
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}"
  echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset}"
  echo "PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-unset} (run.py sets 1 if unset)"
  echo "pvfc imported from: $(cd /tmp && "$PY" -c 'import pvfc; print(pvfc.__file__)' 2>&1)"
  echo "repository: $REPO"
  echo "HEAD: $(git -C "$REPO" rev-parse HEAD) ($(git -C "$REPO" branch --show-current))"
  if [ -z "$(git -C "$REPO" status --porcelain -- src experiments configs pyproject.toml)" ]; then
    echo "src experiments configs pyproject.toml vs HEAD: clean"
  else
    echo "src experiments configs pyproject.toml vs HEAD: DIRTY"
  fi
  echo "git status --short:"; git -C "$REPO" status --short | sed 's/^/  /'
  echo "config: $CONFIG sha256 $(shasum -a 256 -- "$CONFIG" | cut -d' ' -f1)"
  echo "env $ENVNAME vs $LOCK:"
  if [ -f "$LOCK" ] && diff <("$CONDA" env export -n "$ENVNAME" --no-builds 2>/dev/null \
                                | grep -v -E '^(name|prefix):') \
                             <(grep -v -E '^(name|prefix):' "$LOCK") >/dev/null; then
    echo "  identical"
  else
    echo "  DIFFERS"
  fi
  echo "package versions:"
  "$PY" - <<'EOF' 2>&1 | sed 's/^/  /'
from importlib.metadata import PackageNotFoundError, version
for name in ("torch", "timesfm", "tirex-2", "xlstm", "flashrnn", "darts", "chronos-forecasting",
             "huggingface_hub"):
    try:
        print(f"{name} {version(name)}")
    except PackageNotFoundError:
        print(f"{name} NOT INSTALLED")
EOF
  echo "checkpoint revisions (refs/main) in $HUB:"
  for r in "$HUB"/models--*; do
    case $r in
      *chronos-2*|*timesfm*|*TiRex*)
        echo "  $(basename "$r"): refs/main $(cat "$r/refs/main" 2>/dev/null || echo none);" \
             "snapshots $(ls "$r/snapshots" 2>/dev/null | tr '\n' ' ')";;
    esac
  done
  echo "Chronos-2 refs/main (what darts loads) against the pins:"
  pin models--amazon--chronos-2 "$CHRONOS2"
  pin models--autogluon--chronos-2-small "$CHRONOS2_SMALL"
  echo "weights (sha256):"
  weight "$TIMESFM3/model.safetensors" "$SHA_TIMESFM3"
  weight "$TIREX2/model.ckpt" "$SHA_TIREX2"
  echo "swap: $(sysctl -n vm.swapusage)"
  echo "memory pressure: $(memory_pressure 2>/dev/null | tail -1)"
} > "$OUT" 2>&1
cat "$OUT"
