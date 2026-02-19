#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_VIDEO="${AUTORISK_INPUT_VIDEO:-data/public_samples/uk_dashcam_compilation.mp4}"
RUN_DIR="${AUTORISK_RUN_DIR:-outputs/public_run}"
HANDOFF_OUT="${AUTORISK_HANDOFF_OUT:-${RUN_DIR}/handoff_latest}"
CONFIG_PATH="${AUTORISK_CONFIG_PATH:-configs/public.yaml}"
POLICY_PATH="${AUTORISK_POLICY_PATH:-configs/policy.yaml}"
KEYROOT="${AUTORISK_KEY_DIR:-artifacts/judge_keys}"
PRIVATE_KEY="${KEYROOT}/private.pem"
TRUSTED_DIR="${KEYROOT}/trusted"
PUBLIC_KEY="${TRUSTED_DIR}/active.pem"

mkdir -p "${TRUSTED_DIR}"

if [[ ! -f "${PRIVATE_KEY}" || ! -f "${PUBLIC_KEY}" ]]; then
  python - <<'PY'
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

root = Path("artifacts/judge_keys")
trusted = root / "trusted"
root.mkdir(parents=True, exist_ok=True)
trusted.mkdir(parents=True, exist_ok=True)

private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()

(root / "private.pem").write_bytes(
    private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
)
(trusted / "active.pem").write_bytes(
    public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
)
print("Generated signing keypair under artifacts/judge_keys")
PY
fi

if [[ ! -f "${RUN_DIR}/cosmos_results.json" ]]; then
  if [[ ! -f "${INPUT_VIDEO}" ]]; then
    echo "Error: input video not found: ${INPUT_VIDEO}" >&2
    echo "Set AUTORISK_INPUT_VIDEO to a rights-cleared local video path and retry." >&2
    exit 2
  fi
  python -m autorisk.cli -c "${CONFIG_PATH}" run -i "${INPUT_VIDEO}" -o "${RUN_DIR}"
fi

python -m autorisk.cli finalize-run \
  -r "${RUN_DIR}" \
  --policy "${POLICY_PATH}" \
  --zip \
  --audit-grade \
  --sign-private-key "${PRIVATE_KEY}" \
  --sign-public-key-dir "${TRUSTED_DIR}" \
  --handoff-out "${HANDOFF_OUT}" \
  --enforce

python -m autorisk.cli audit-verify \
  -p "${HANDOFF_OUT}/PACK.zip" \
  --profile audit-grade \
  --public-key-dir "${TRUSTED_DIR}" \
  --json-out "${HANDOFF_OUT}/verify_result.json"

python -m autorisk.cli audit-handoff-verify \
  -d "${HANDOFF_OUT}" \
  --profile audit-grade \
  --enforce \
  --json-out "${HANDOFF_OUT}/handoff_verify_result.json"

export AUTORISK_HANDOFF_OUT_RESOLVED="${HANDOFF_OUT}"
python - <<'PY'
import json
import os
from pathlib import Path

handoff_out = Path(os.environ["AUTORISK_HANDOFF_OUT_RESOLVED"]).resolve()
verify_json = handoff_out / "verify_result.json"
payload = json.loads(verify_json.read_text(encoding="utf-8"))
fingerprint = payload.get("checksums_sha256", "")
print("")
print("Judge artifacts ready:")
print(f"- Handoff dir: {handoff_out}")
print(f"- Pack zip:    {handoff_out / 'PACK.zip'}")
print(f"- Fingerprint: {fingerprint}")
print("")
print("Use this fingerprint for external ticket/DB pinning.")
PY
