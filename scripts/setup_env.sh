#!/usr/bin/env bash
# One-shot environment setup. Creates .venv, installs deps (torch resolves to
# the right CUDA/MPS/CPU build for THIS machine), freezes the lockfile, and
# smoke-tests the GPU + timm model names before anything trains.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements.lock.txt
echo "Froze exact versions to requirements.lock.txt — commit it."

python - <<'EOF'
import platform, torch, timm

print(f"python {platform.python_version()}  torch {torch.__version__}  timm {timm.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}  (CUDA {torch.version.cuda})")
elif torch.backends.mps.is_available():
    print("Apple MPS available — fine for smoke tests; real training belongs on the CUDA machine")
else:
    print("WARNING: CPU only — smoke tests only")

# Rule: verify timm model names on the actual install instead of trusting memory
expected = [
    "resnet50.tv2_in1k",
    "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2_base.fcmae_ft_in22k_in1k_384",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "tf_efficientnetv2_m.in21k_ft_in1k",
    "eva02_base_patch14_448.mim_in22k_ft_in1k",
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
]
available = set(timm.list_models(pretrained=True))
for name in expected:
    status = "OK" if name in available else "MISSING — fix the config before training"
    print(f"  {name:55s} {status}")
EOF

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
