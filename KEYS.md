# Key Management (Audit Signing)

This document defines key handling for `audit-sign` / `audit-verify`.

## Scope

- Signing algorithm: `Ed25519`
- Signed payload: SHA256 of `checksums.sha256.txt` and `manifest.json`
- Signature artifact: `signature.json` in the audit pack root

## Trust Model

- Integrity is provided by `checksums.sha256.txt`.
- Authenticity is provided only when verification uses a trusted external public key:
  - `python -m autorisk.cli audit-verify -p PACK --public-key keys/public.pem --require-signature --require-public-key`
- Embedded public keys (`public_key_pem` in `signature.json`) are convenience-only and are not trusted by default.
- Audit-grade operational entrypoint:
  - `python -m autorisk.cli finalize-run -r RUN_DIR --policy configs/policy.yaml --zip --audit-grade --sign-private-key keys/private.pem --sign-public-key keys/public.pem`

## Private Key Handling

- Never commit private keys to git.
- Store private keys in a secure secret store (HSM/KMS or encrypted secrets manager).
- Restrict signing access to approved CI/service accounts.
- Use short-lived credentials for CI jobs that invoke signing.

## Public Key Distribution

- Distribute trusted public keys out-of-band (security channel, artifact registry, or governance repo).
- Consumers must pin the expected public key file and verify key provenance before use.
- For key rotation, publish multiple trusted public keys and verify with:
  - `python -m autorisk.cli audit-verify -p PACK --public-key-dir keys/trusted --require-signature --require-public-key`
- To share verifier assets with third parties, build a verifier bundle:
  - `python -m autorisk.cli audit-verifier-bundle --out verifier_bundle --public-key-dir keys/trusted --revocation-file revoked_key_ids.txt`

## Key Rotation

1. Generate a new Ed25519 key pair.
2. Publish the new public key to trusted distribution channels.
3. Update signing jobs to use the new private key.
4. Keep previous public keys available during migration.
5. Record rotation date and owner in internal security logs.

## Key ID Semantics

- `key_id` is derived from the public key (`sha256(raw_public_key)[:16]`).
- `key_id` is not user-overridable.
- Optional `key_label` is metadata only and can be set at signing time.

## Revocation / Incident Response

- If key compromise is suspected:
  - Immediately stop signing with the compromised key.
  - Mark the corresponding public key as revoked in trusted distribution channels.
  - Rotate keys and re-issue signed packs as needed.
  - During verification, reject revoked `key_id` values by policy:
    - `--revoked-key-id <id>` and/or `--revocation-file revoked_key_ids.txt`

## Encrypted Private Keys

- Encrypted PEM private keys are supported.
- Prefer password delivery via environment variable:
  - `--private-key-password-env AUTORISK_SIGNING_KEY_PASSWORD`
- Avoid plain `--private-key-password` in shared shells or CI logs.
