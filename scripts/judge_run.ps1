param(
    [string]$InputVideo = "",
    [string]$RunDir = "outputs/public_run",
    [string]$HandoffOut = "",
    [string]$ConfigPath = "configs/public.yaml",
    [string]$PolicyPath = "configs/policy.yaml",
    [string]$KeyRoot = "artifacts/judge_keys"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($InputVideo)) {
    if (-not [string]::IsNullOrWhiteSpace($env:AUTORISK_INPUT_VIDEO)) {
        $InputVideo = $env:AUTORISK_INPUT_VIDEO
    } else {
        $InputVideo = "data/public_samples/uk_dashcam_compilation.mp4"
    }
}
if ([string]::IsNullOrWhiteSpace($HandoffOut)) {
    $HandoffOut = Join-Path $RunDir "handoff_latest"
}

$privateKey = Join-Path $KeyRoot "private.pem"
$trustedDir = Join-Path $KeyRoot "trusted"
$publicKey = Join-Path $trustedDir "active.pem"

New-Item -ItemType Directory -Force -Path $trustedDir | Out-Null

if (-not (Test-Path $privateKey) -or -not (Test-Path $publicKey)) {
    @'
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
'@ | python -
}

$resultsPath = Join-Path $RunDir "cosmos_results.json"
if (-not (Test-Path $resultsPath)) {
    if (-not (Test-Path $InputVideo)) {
        Write-Error "Input video not found: $InputVideo`nSet AUTORISK_INPUT_VIDEO to a rights-cleared local video path and retry."
    }
    python -m autorisk.cli -c $ConfigPath run -i $InputVideo -o $RunDir
}

python -m autorisk.cli finalize-run `
    -r $RunDir `
    --policy $PolicyPath `
    --zip `
    --audit-grade `
    --sign-private-key $privateKey `
    --sign-public-key-dir $trustedDir `
    --handoff-out $HandoffOut `
    --enforce

python -m autorisk.cli audit-verify `
    -p (Join-Path $HandoffOut "PACK.zip") `
    --profile audit-grade `
    --public-key-dir $trustedDir `
    --json-out (Join-Path $HandoffOut "verify_result.json")

python -m autorisk.cli audit-handoff-verify `
    -d $HandoffOut `
    --profile audit-grade `
    --enforce `
    --json-out (Join-Path $HandoffOut "handoff_verify_result.json")

$verifyJsonPath = Join-Path $HandoffOut "verify_result.json"
$verify = Get-Content $verifyJsonPath -Raw | ConvertFrom-Json
$fingerprint = [string]$verify.checksums_sha256

Write-Host ""
Write-Host "Judge artifacts ready:"
Write-Host "- Handoff dir: $((Resolve-Path $HandoffOut).Path)"
Write-Host "- Pack zip:    $((Resolve-Path (Join-Path $HandoffOut 'PACK.zip')).Path)"
Write-Host "- Fingerprint: $fingerprint"
Write-Host ""
Write-Host "Use this fingerprint for external ticket/DB pinning."
