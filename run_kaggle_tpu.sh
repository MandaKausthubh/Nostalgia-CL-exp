#!/usr/bin/env bash
# Kaggle TPU v3-8 runner for the DomainNet CL sweep.
#
# Kaggle differences from the RunPod path this script handles:
#   * /kaggle/input is READ-ONLY -> wandb must go under /kaggle/working.
#   * /kaggle/working has a ~20 GB output cap.
#   * torch_xla is NOT preinstalled -> INSTALL_DEPS=1 pip-installs torch_xla and
#     pytorch-adapt (DomainNet's list-file loader needs the latter).
#   * One process drives all 8 chips (Lightning XLA strategy). Runs are serial.
#
# Usage (from a Kaggle notebook cell):
#   !bash /kaggle/working/Nostalgia-CL-exp/run_kaggle_tpu.sh
#   MODE=smoke !bash .../run_kaggle_tpu.sh          # 2 domains, 2 methods, tiny
#
# Override any axis with env: SEEDS, METHODS, BACKBONES, PH2, K, LORA_R, ...

set -euo pipefail

# ----- Paths -------------------------------------------------------------
REPO_DIR="${REPO_DIR:-/kaggle/working/Nostalgia-CL-exp}"
DATA_ROOT_DN="${DATA_ROOT_DN:-/kaggle/input/datasets/kausthubhmanda/domainnet-fulldataset/domainnet}"
WANDB_DIR="${WANDB_DIR:-/kaggle/working/wandb_logs}"

# ----- Accelerator -------------------------------------------------------
export ACCEL="${ACCEL:-tpu}"
export DEVICES="${DEVICES:-8}"          # v3-8 -> 8 chips, single process
export STRATEGY="${STRATEGY:-xla}"
export PRECISION="${PRECISION:-bf16-true}"   # NOT bf16-mixed on XLA
export NUM_WORKERS="${NUM_WORKERS:-0}"       # ignored off-CUDA anyway

# ----- Run policy --------------------------------------------------------
# Kaggle TPU VMs usually have internet, but keep wandb local-first. Set
# WANDB_MODE=online to stream to the dashboard.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Kaggle exposes no shell env for secrets, so pull the wandb key from the
# notebook's Secrets store (Add-ons > Secrets) instead of pasting a credential
# into a cell. Tries a few labels; no-op when WANDB_API_KEY is already set or
# the store is unavailable. Without a key, wandb "online" would just fail on
# login, so fall back to offline rather than lose the run.
if [ "$WANDB_MODE" = "online" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    _key="$(python - <<'PY'
for label in ("WANDB_API_KEY", "WANDB_KEY", "wandb-api-key"):
    try:
        from kaggle_secrets import UserSecretsClient
        val = UserSecretsClient().get_secret(label)
    except Exception:
        val = None
    if val:
        print(val.strip())
        break
PY
)"
    if [ -n "$_key" ]; then
        export WANDB_API_KEY="$_key"
        echo "  [ok] WANDB_API_KEY loaded from Kaggle Secrets"
    else
        echo "  [warn] WANDB_MODE=online but no key in Kaggle Secrets."
        echo "         Add secret 'WANDB_API_KEY' (Add-ons > Secrets), then re-run."
        echo "         Falling back to WANDB_MODE=offline."
        export WANDB_MODE=offline
    fi
    unset _key
fi

export DATA_ROOT_DN WANDB_DIR

MODE="${MODE:-full}"   # full | smoke

echo "====================================================================="
echo "Kaggle TPU runner"
echo "  REPO_DIR         = $REPO_DIR"
echo "  DATA_ROOT_DN     = $DATA_ROOT_DN"
echo "  WANDB_DIR        = $WANDB_DIR"
echo "  MODE             = $MODE"
echo "  ACCEL/DEVICES    = $ACCEL / $DEVICES   (strategy=$STRATEGY, precision=$PRECISION)"
echo "  WANDB_MODE       = $WANDB_MODE"
echo "====================================================================="

# ----- 1. Deps -----------------------------------------------------------
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    echo "=== [1/4] Deps ==="
    python -m pip install -q --upgrade pip

    # Python 3.12 no longer ships setuptools in the venv, but torchmetrics
    # (pulled in by Lightning) does `from pkg_resources import ...` at import
    # time -> ModuleNotFoundError: No module named 'pkg_resources'.
    # setuptools 82.0.0 (Feb 2026) REMOVED pkg_resources entirely, and an
    # unqualified `pip install setuptools` grabs that latest -> still broken.
    # Pin <82 so the bundled pkg_resources survives.
    python -m pip install -q "setuptools<82"
    python -c "import pkg_resources" 2>/dev/null || {
        echo "[FATAL] pkg_resources still missing after installing setuptools<82."
        echo "        torchmetrics/Lightning need it. Check pip target:"
        python -c "import site,sys; print(sys.executable); print(site.getsitepackages())"
        exit 1
    }

    # torch_xla must share torch's ABI. A mismatched wheel dies at import with
    #   undefined symbol: _ZN5torch8autograd13_wrap_outputsE...
    # Two gotchas, both seen on Kaggle (torch 2.10.0+cpu):
    #   1. torch_xla LAGS torch on PyPI (newest is 2.9.0 when torch is 2.10) and
    #      does NOT declare torch as a dependency, so pip never realigns it.
    #   2. torch_xla's [tpu] extra pulls libtpu, whose wheels live on the Google
    #      release index (--find-links), not PyPI.
    # So: pick the newest torch_xla <= torch, install it + the *matching*
    # torch/torchvision explicitly. Override with TORCH_XLA_VERSION=<ver>.
    TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
    LIBTPU_INDEX="https://storage.googleapis.com/libtpu-releases/index.html"

    # torch -> torchvision (kept consistent so other Kaggle packages still
    # import; unknown pairs are skipped with a warning).
    tv_for() {
        case "$1" in
            2.10.*) echo "0.25.0" ;;
            2.9.*)  echo "0.24.0" ;;
            2.8.*)  echo "0.23.0" ;;
            2.7.*)  echo "0.22.0" ;;
            2.6.*)  echo "0.21.0" ;;
            2.5.*)  echo "0.20.0" ;;
            *)      echo "" ;;
        esac
    }

    if ! python -c "import torch_xla" >/dev/null 2>&1; then
        XLA_VER="${TORCH_XLA_VERSION:-}"
        if [ -z "$XLA_VER" ]; then
            # Newest non-prerelease torch_xla <= torch's version.
            XLA_VER="$(python -c '
import json, sys, urllib.request
want = tuple(int(x) for x in sys.argv[1].split(".")[:3])
try:
    rel = json.load(urllib.request.urlopen(
        "https://pypi.org/pypi/torch_xla/json", timeout=30))["releases"]
except Exception:
    print(sys.argv[1]); raise SystemExit
def k(v):
    p = v.split(".")
    return tuple(int(x) for x in p[:3]) if len(p) >= 3 and all(x.isdigit() for x in p[:3]) else None
c = [v for v in rel if rel[v] and not any(ch.isalpha() for ch in v) and k(v) and k(v) <= want]
print(max(c, key=k) if c else sys.argv[1])
' "$TORCH_VER")"
        fi

        echo "  torch=$TORCH_VER -> torch_xla[tpu]==$XLA_VER"
        [ "$XLA_VER" != "$TORCH_VER" ] && \
            echo "  (torch_xla lags torch; will realign torch to $XLA_VER)"

        # Drop any mismatched wheel a previous run left behind.
        python -m pip uninstall -y torch_xla >/dev/null 2>&1 || true

        if ! python -m pip install -q "torch_xla[tpu]==${XLA_VER}" -f "$LIBTPU_INDEX"; then
            echo "[FATAL] no torch_xla[tpu]==${XLA_VER} wheel available."
            echo "        torch is ${TORCH_VER}; pick a matching pair, e.g."
            echo "          pip install torch==2.5.0 torchvision==0.20.0 'torch_xla[tpu]==2.5.0' -f $LIBTPU_INDEX"
            echo "        or set TORCH_XLA_VERSION=<ver>."
            exit 1
        fi

        # torch_xla does not pin torch, so realign torch + its siblings ourselves
        # when the chosen wheel's version differs from what the image ships.
        # torchaudio MUST be realigned too: transformers imports it (via
        # loss_rnnt) and a stale build dies with
        #   libtorchaudio.so: undefined symbol: ...c10::SymInt::sym_ne...
        # torchaudio's version tracks torch's exactly (2.9.0 <-> 2.9.0).
        if [ "$XLA_VER" != "$TORCH_VER" ]; then
            TV_VER="$(tv_for "$XLA_VER")"
            realign=("torch==${XLA_VER}" "torchaudio==${XLA_VER}")
            [ -n "$TV_VER" ] && realign+=("torchvision==${TV_VER}")
            echo "  realigning ${realign[*]} ..."
            if ! python -m pip install -q "${realign[@]}"; then
                echo "  [warn] realign failed; pinning torch only"
                python -m pip install -q "torch==${XLA_VER}" || true
            fi
        fi
    fi

    # DomainNet loader imports pytorch_adapt.datasets.
    if ! python -c "import pytorch_adapt" >/dev/null 2>&1; then
        echo "  installing pytorch-adapt ..."
        python -m pip install -q pytorch-adapt
    fi

    python - <<'PY'
import torch
print(f"  torch = {torch.__version__}")
try:
    import torch_xla
    import torch_xla.runtime as xr
    print(f"  torch_xla = {torch_xla.__version__}")
    print(f"  xla device count = {xr.global_device_count()}")
except Exception as exc:  # noqa: BLE001
    print(f"  [FATAL] torch_xla unavailable: {exc}")
    raise SystemExit(1)
PY
else
    echo "=== [1/4] Deps skipped (INSTALL_DEPS=0) ==="
fi

# ----- 2. Writable dirs --------------------------------------------------
echo "=== [2/4] Writable dirs ==="
for d in "$WANDB_DIR"; do
    mkdir -p "$d"
    if ! touch "$d/.write_test" 2>/dev/null; then
        echo "[FATAL] $d is not writable"
        exit 1
    fi
    rm -f "$d/.write_test"
    echo "  [ok] $d"
done

# ----- 3. Sanity: repo + dataset layout ----------------------------------
echo "=== [3/4] Sanity ==="
[ -d "$REPO_DIR" ] || { echo "[FATAL] repo not found at $REPO_DIR"; exit 1; }
[ -d "$DATA_ROOT_DN" ] || { echo "[FATAL] dataset not found at $DATA_ROOT_DN"; exit 1; }

# The loader (pytorch_adapt) expects <root>/domainnet/{domain}_{train,test}.txt.
# It strips a trailing 'domainnet' basename, so either the parent or the
# domainnet dir itself is acceptable here.
_probe="$DATA_ROOT_DN"
if [ ! -d "$_probe/clipart" ] && [ -d "$_probe/domainnet/clipart" ]; then
    _probe="$_probe/domainnet"
fi
for d in clipart infograph painting quickdraw real sketch; do
    if [ ! -d "$_probe/$d" ] || [ ! -f "$_probe/${d}_train.txt" ]; then
        echo "[FATAL] missing $_probe/$d (or ${d}_train.txt)"
        exit 1
    fi
done
echo "  [ok] 6 domain folders + split lists present under $_probe"

# Read-only input is expected; only warn if we resolved to something else.
case "$DATA_ROOT_DN" in
    /kaggle/input/*) echo "  [ok] dataset under read-only /kaggle/input (expected)" ;;
    *) echo "  [warn] DATA_ROOT_DN is not under /kaggle/input: $DATA_ROOT_DN" ;;
esac

DF_OUT="$(df -h /kaggle/working 2>/dev/null | tail -1 || true)"
[ -n "$DF_OUT" ] && echo "  /kaggle/working: $DF_OUT  (~20 GB output cap)"

# ----- 4. Launch sweep ---------------------------------------------------
echo "=== [4/4] Sweep ==="
cd "$REPO_DIR"

if [ "$MODE" = "smoke" ]; then
    echo "  smoke: 2 domains, nostalgia + naive_adam, resnet18, tiny budget"
    TASKS="domainnet_clipart domainnet_real" \
    SEEDS="0" \
    BACKBONES="resnet18" \
    METHODS="nostalgia naive_adam" \
    PH1=1 PH2=1 WARMUP=10 TOTAL_STEPS=50 VAL_EPOCHS=1 \
    bash run_domainnet_sweep.sh
else
    bash run_domainnet_sweep.sh
fi

echo "Kaggle TPU run finished."
