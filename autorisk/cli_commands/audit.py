"""Audit/review/policy/finalize CLI command registrations."""

from __future__ import annotations

import os
from pathlib import Path

import click


def _resolve_private_key_password(
    *,
    private_key_password: str | None,
    private_key_password_env: str | None,
) -> str | None:
    if private_key_password is not None and private_key_password_env is not None:
        raise click.UsageError(
            "Use either --private-key-password or --private-key-password-env, not both."
        )
    if private_key_password_env is not None:
        env_name = str(private_key_password_env).strip()
        if env_name == "":
            raise click.UsageError("--private-key-password-env cannot be empty.")
        value = os.environ.get(env_name)
        if value is None:
            raise click.UsageError(f"Environment variable not set: {env_name}")
        return value
    if private_key_password is None:
        return None
    return str(private_key_password)


def _load_revoked_key_ids(
    *,
    revoked_key_ids: tuple[str, ...],
    revocation_file: str | None,
) -> set[str]:
    out = {str(v).strip().lower() for v in revoked_key_ids if str(v).strip() != ""}
    if revocation_file is None or str(revocation_file).strip() == "":
        return out
    path = Path(revocation_file).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise click.UsageError(f"Revocation file not found: {path}")
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        out.add(line.split()[0].lower())
    return out


def _format_effective_flags(flags: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in flags.items():
        if isinstance(value, bool):
            parts.append(f"{key}={'true' if value else 'false'}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


@click.command("audit-pack")
@click.option(
    "--run-dir",
    "-r",
    required=True,
    help="Run directory containing cosmos_results.json and candidates.csv",
)
@click.option(
    "--input-video",
    "-i",
    default=None,
    help="Optional source input video path (for SHA256 provenance)",
)
@click.option(
    "--review-log",
    default=None,
    help="Optional human review log jsonl to include in the pack",
)
@click.option(
    "--out",
    "-o",
    "output_dir",
    default=None,
    help="Output directory (default: RUN_DIR/audit_pack_<timestamp>)",
)
@click.option(
    "--include-clips/--no-include-clips",
    default=True,
    show_default=True,
    help="Copy candidate clips into the pack",
)
@click.option(
    "--zip/--no-zip",
    "create_zip",
    default=True,
    show_default=True,
    help="Create zip bundle for handoff",
)
@click.pass_context
def audit_pack(
    ctx: click.Context,
    run_dir: str,
    input_video: str | None,
    review_log: str | None,
    output_dir: str | None,
    include_clips: bool,
    create_zip: bool,
) -> None:
    """Build an auditable evidence pack (manifest + decision trace + checksums)."""
    from autorisk.audit.pack import build_audit_pack

    cfg = (ctx.obj or {}).get("cfg")
    result = build_audit_pack(
        run_dir=run_dir,
        cfg=cfg,
        output_dir=output_dir,
        input_video=input_video,
        review_log=review_log,
        include_clips=include_clips,
        create_zip=create_zip,
    )

    click.echo(f"Audit pack directory: {result.output_dir}")
    click.echo(f"Records: {result.records}")
    click.echo(f"Manifest: {result.manifest_path}")
    click.echo(f"Decision trace: {result.decision_trace_path}")
    click.echo(f"Checksums: {result.checksums_path}")
    click.echo(f"Pack fingerprint (checksums SHA256): {result.checksums_sha256}")
    if result.zip_path is not None:
        click.echo(f"Zip: {result.zip_path}")


@click.command("audit-sign")
@click.option("--pack", "-p", required=True, help="Audit pack directory or zip bundle")
@click.option("--private-key", required=True, help="Ed25519 private key PEM path")
@click.option(
    "--private-key-password",
    default=None,
    help="Private key password (less secure; prefer --private-key-password-env)",
)
@click.option(
    "--private-key-password-env",
    default=None,
    help="Environment variable containing private key password",
)
@click.option("--public-key", default=None, help="Optional Ed25519 public key PEM path")
@click.option(
    "--key-label", default=None, help="Optional key label metadata (for audit logs)"
)
@click.option(
    "--embed-public-key/--no-embed-public-key",
    default=False,
    show_default=True,
    help="Embed public key PEM in signature.json",
)
def audit_sign(
    pack: str,
    private_key: str,
    private_key_password: str | None,
    private_key_password_env: str | None,
    public_key: str | None,
    key_label: str | None,
    embed_public_key: bool,
) -> None:
    """Sign an audit pack using Ed25519."""
    from autorisk.audit.sign import sign_audit_pack

    resolved_password = _resolve_private_key_password(
        private_key_password=private_key_password,
        private_key_password_env=private_key_password_env,
    )
    res = sign_audit_pack(
        pack,
        private_key_path=private_key,
        private_key_password=resolved_password,
        public_key_path=public_key,
        key_label=key_label,
        include_public_key=embed_public_key,
    )
    click.echo(f"Source: {res.source}")
    click.echo(f"Mode: {res.mode}")
    click.echo(f"Signature: {res.signature_path}")
    click.echo(f"Key ID: {res.key_id}")
    click.echo(f"Checksums SHA256: {res.checksums_sha256}")
    click.echo(f"Manifest SHA256: {res.manifest_sha256}")


@click.command("audit-attest")
@click.option("--pack", "-p", required=True, help="Audit pack directory or zip bundle")
@click.option("--private-key", required=True, help="Ed25519 private key PEM path")
@click.option(
    "--private-key-password",
    default=None,
    help="Private key password (less secure; prefer --private-key-password-env)",
)
@click.option(
    "--private-key-password-env",
    default=None,
    help="Environment variable containing private key password",
)
@click.option("--public-key", default=None, help="Optional Ed25519 public key PEM path")
@click.option(
    "--key-label",
    default=None,
    help="Optional human-readable key label (metadata only)",
)
@click.option(
    "--embed-public-key/--no-embed-public-key",
    default=False,
    show_default=True,
    help="Embed public key PEM in attestation.json",
)
def audit_attest(
    pack: str,
    private_key: str,
    private_key_password: str | None,
    private_key_password_env: str | None,
    public_key: str | None,
    key_label: str | None,
    embed_public_key: bool,
) -> None:
    """Generate attestation.json over non-checksummed run artifacts."""
    from autorisk.audit.attestation import attest_audit_pack

    resolved_password = _resolve_private_key_password(
        private_key_password=private_key_password,
        private_key_password_env=private_key_password_env,
    )
    res = attest_audit_pack(
        pack,
        private_key_path=private_key,
        private_key_password=resolved_password,
        public_key_path=public_key,
        key_label=key_label,
        include_public_key=embed_public_key,
    )
    click.echo(f"Source: {res.source}")
    click.echo(f"Mode: {res.mode}")
    click.echo(f"Attestation: {res.attestation_path}")
    click.echo(f"Key ID: {res.key_id}")
    click.echo(f"Pack fingerprint: {res.pack_fingerprint}")
    click.echo(f"Finalize record SHA256: {res.finalize_record_sha256}")
    click.echo(f"Validate report SHA256: {res.audit_validate_report_sha256}")


@click.command("audit-verify")
@click.option("--pack", "-p", required=True, help="Audit pack directory or zip bundle")
@click.option(
    "--profile",
    type=click.Choice(["default", "audit-grade"], case_sensitive=False),
    default="default",
    show_default=True,
    help="Verification profile (default=diagnostics, audit-grade=enforced trusted signature + attestation)",
)
@click.option(
    "--strict/--no-strict",
    default=True,
    show_default=True,
    help="Strict verification (fail on any issue)",
)
@click.option(
    "--public-key",
    default=None,
    help="Optional Ed25519 public key PEM path for signature verification",
)
@click.option(
    "--public-key-dir",
    default=None,
    help="Optional trusted public key directory (.pem). Key is selected by signature key_id.",
)
@click.option(
    "--require-signature/--no-require-signature",
    default=False,
    show_default=True,
    help="Fail if signature.json is missing",
)
@click.option(
    "--require-public-key/--no-require-public-key",
    default=False,
    show_default=True,
    help="Fail unless --public-key or --public-key-dir is provided",
)
@click.option(
    "--require-attestation/--no-require-attestation",
    default=False,
    show_default=True,
    help="Fail if attestation.json is missing or unverifiable (recommended for audit decisions)",
)
@click.option(
    "--trust-embedded-public-key/--no-trust-embedded-public-key",
    default=False,
    show_default=True,
    help="Allow verification against public_key_pem embedded in signature.json/attestation.json",
)
@click.option(
    "--revoked-key-id",
    "revoked_key_ids",
    multiple=True,
    help="Revoked signature key_id (repeatable)",
)
@click.option(
    "--revocation-file",
    default=None,
    help="Path to revoked key IDs file (one key_id per line)",
)
@click.option(
    "--expect-pack-fingerprint",
    default=None,
    help="Expected checksums.sha256.txt SHA256 (64 hex) for anti-substitution checks",
)
@click.option(
    "--json-out", default=None, help="Optional path to write verification result JSON"
)
def audit_verify(
    pack: str,
    profile: str,
    strict: bool,
    public_key: str | None,
    public_key_dir: str | None,
    require_signature: bool,
    require_public_key: bool,
    require_attestation: bool,
    trust_embedded_public_key: bool,
    revoked_key_ids: tuple[str, ...],
    revocation_file: str | None,
    expect_pack_fingerprint: str | None,
    json_out: str | None,
) -> None:
    """Verify an audit pack using checksums.sha256.txt."""
    from autorisk.services.audit_verify_service import (
        AuditVerifyServiceRequest,
        run_audit_verify,
    )

    revoked = _load_revoked_key_ids(
        revoked_key_ids=revoked_key_ids,
        revocation_file=revocation_file,
    )
    request = AuditVerifyServiceRequest(
        pack=pack,
        profile=profile,
        strict=bool(strict),
        public_key=public_key,
        public_key_dir=public_key_dir,
        require_signature=bool(require_signature),
        require_public_key=bool(require_public_key),
        require_attestation=bool(require_attestation),
        trust_embedded_public_key=bool(trust_embedded_public_key),
        revoked_key_ids=revoked,
        expect_pack_fingerprint=expect_pack_fingerprint,
        json_out=json_out,
    )
    run_audit_verify(
        request,
        emit=lambda message, is_err: click.echo(message, err=is_err),
    )


@click.command("audit-verifier-bundle")
@click.option("--out", required=True, help="Output directory for verifier bundle")
@click.option("--public-key", default=None, help="Trusted public key PEM path")
@click.option(
    "--public-key-dir", default=None, help="Trusted public key directory (.pem)"
)
@click.option(
    "--revoked-key-id",
    "revoked_key_ids",
    multiple=True,
    help="Revoked signature key_id (repeatable)",
)
@click.option(
    "--revocation-file",
    default=None,
    help="Path to revoked key IDs file (one key_id per line)",
)
@click.option(
    "--pack-ref", default="PACK_OR_ZIP", help="Pack path placeholder shown in VERIFY.md"
)
def audit_verifier_bundle(
    out: str,
    public_key: str | None,
    public_key_dir: str | None,
    revoked_key_ids: tuple[str, ...],
    revocation_file: str | None,
    pack_ref: str,
) -> None:
    """Build verifier bundle (trusted keys + revocation file + VERIFY.md)."""
    from autorisk.audit.verifier_bundle import build_verifier_bundle

    revoked = _load_revoked_key_ids(
        revoked_key_ids=revoked_key_ids,
        revocation_file=revocation_file,
    )
    res = build_verifier_bundle(
        output_dir=out,
        public_key=public_key,
        public_key_dir=public_key_dir,
        revoked_key_ids=revoked,
        revocation_file=revocation_file,
        verify_pack_reference=pack_ref,
    )
    click.echo(f"Verifier bundle: {res.output_dir}")
    click.echo(f"Trusted keys dir: {res.trusted_keys_dir}")
    click.echo(f"Trusted keys: {len(res.key_files)}")
    click.echo(f"Revocation file: {res.revocation_path}")
    click.echo(f"Revoked key IDs: {len(res.revoked_key_ids)}")
    click.echo(f"VERIFY: {res.verify_md_path}")


@click.command("audit-validate")
@click.option("--pack", "-p", required=True, help="Audit pack directory or zip bundle")
@click.option(
    "--schema-dir", default=None, help="Schema directory (default: repo/schemas)"
)
@click.option(
    "--profile",
    type=click.Choice(["default", "audit-grade"], case_sensitive=False),
    default="default",
    show_default=True,
    help="Validation profile (audit-grade enforces required handoff artifacts)",
)
@click.option(
    "--require-signature/--no-require-signature",
    default=False,
    show_default=True,
    help="Require signature.json",
)
@click.option(
    "--require-finalize-record/--no-require-finalize-record",
    default=False,
    show_default=True,
    help="Require run_artifacts/finalize_record.json",
)
@click.option(
    "--require-validate-report/--no-require-validate-report",
    default=False,
    show_default=True,
    help="Require run_artifacts/audit_validate_report.json",
)
@click.option(
    "--require-policy-snapshot/--no-require-policy-snapshot",
    default=False,
    show_default=True,
    help="Require run_artifacts/policy_snapshot.json",
)
@click.option(
    "--require-review-artifacts/--no-require-review-artifacts",
    default=False,
    show_default=True,
    help="Require review artifacts (review_apply/review_diff/reviewed_results)",
)
@click.option(
    "--semantic/--no-semantic",
    "semantic_checks",
    default=True,
    show_default=True,
    help="Run semantic consistency checks in addition to JSON Schema validation",
)
@click.option(
    "--enforce/--no-enforce",
    default=False,
    show_default=True,
    help="Exit non-zero when any validation issue is found",
)
@click.option(
    "--json-out", default=None, help="Optional path to write validation result JSON"
)
def audit_validate(
    pack: str,
    schema_dir: str | None,
    profile: str,
    require_signature: bool,
    require_finalize_record: bool,
    require_validate_report: bool,
    require_policy_snapshot: bool,
    require_review_artifacts: bool,
    semantic_checks: bool,
    enforce: bool,
    json_out: str | None,
) -> None:
    """Validate audit-pack contract (schema + semantics)."""
    from autorisk.audit.validate import validate_audit_pack

    result = validate_audit_pack(
        pack,
        schema_dir=schema_dir,
        semantic_checks=semantic_checks,
        profile=profile,
        require_signature=require_signature,
        require_finalize_record=require_finalize_record,
        require_validate_report=require_validate_report,
        require_policy_snapshot=require_policy_snapshot,
        require_review_artifacts=require_review_artifacts,
    )

    click.echo(f"Source: {result.source}")
    click.echo(f"Mode: {result.mode}")
    click.echo(f"Pack root: {result.pack_root}")
    click.echo(f"Schema dir: {result.schema_dir}")
    click.echo(f"Files validated: {result.files_validated}")
    click.echo(f"Records validated: {result.records_validated}")
    click.echo(f"Issues: {len(result.issues)}")

    if json_out is not None and str(json_out).strip() != "":
        out_path = Path(json_out).expanduser().resolve()
        out_path.write_text(result.to_json(), encoding="utf-8")
        click.echo(f"Wrote: {out_path}")

    if result.issues:
        for issue in result.issues[:50]:
            line_text = f":{issue.line}" if issue.line is not None else ""
            click.echo(
                f"- {issue.kind}: {issue.path}{line_text} {issue.detail}".rstrip()
            )
        if len(result.issues) > 50:
            click.echo(f"... ({len(result.issues) - 50} more)")
        if enforce:
            raise SystemExit(2)


@click.command("audit-handoff")
@click.option(
    "--run-dir",
    "-r",
    required=True,
    help="Run directory containing finalize-run outputs",
)
@click.option(
    "--out",
    "-o",
    "output_dir",
    default=None,
    help="Output handoff directory (default: RUN_DIR/handoff_<timestamp>)",
)
@click.option(
    "--pack-zip",
    default=None,
    help="Optional audit pack zip path (default: latest RUN_DIR/audit_pack_*.zip)",
)
@click.option(
    "--verifier-bundle-dir",
    default=None,
    help="Optional verifier bundle directory (default: RUN_DIR/verifier_bundle)",
)
@click.option(
    "--finalize-record",
    default=None,
    help="Optional finalize_record.json path (default: RUN_DIR/finalize_record.json)",
)
def audit_handoff(
    run_dir: str,
    output_dir: str | None,
    pack_zip: str | None,
    verifier_bundle_dir: str | None,
    finalize_record: str | None,
) -> None:
    """Build single handoff artifact set (PACK.zip + verifier bundle + metadata)."""
    from autorisk.audit.handoff import build_audit_handoff

    result = build_audit_handoff(
        run_dir=run_dir,
        output_dir=output_dir,
        pack_zip=pack_zip,
        verifier_bundle_dir=verifier_bundle_dir,
        finalize_record=finalize_record,
    )

    click.echo(f"Handoff directory: {result.output_dir}")
    click.echo(f"Pack: {result.pack_zip_path}")
    click.echo(f"Verifier bundle zip: {result.verifier_bundle_zip_path}")
    click.echo(f"Finalize record: {result.finalize_record_path}")
    if result.validate_report_path is not None:
        click.echo(f"Validate report: {result.validate_report_path}")
    click.echo(f"Guide: {result.handoff_guide_path}")
    click.echo(f"Checksums: {result.checksums_path}")


@click.command("audit-handoff-verify")
@click.option(
    "--handoff",
    "-d",
    "handoff_dir",
    required=True,
    help="Handoff directory containing PACK.zip and verifier_bundle.zip",
)
@click.option(
    "--profile",
    type=click.Choice(["default", "audit-grade"], case_sensitive=False),
    default="audit-grade",
    show_default=True,
    help="Verification profile (default=diagnostics, audit-grade=enforced trusted signature + attestation)",
)
@click.option(
    "--strict/--no-strict",
    default=True,
    show_default=True,
    help="Strict checksum verification for bundled PACK.zip",
)
@click.option(
    "--require-signature/--no-require-signature",
    default=True,
    show_default=True,
    help="Require signature.json in bundled PACK.zip",
)
@click.option(
    "--require-public-key/--no-require-public-key",
    default=True,
    show_default=True,
    help="Require trusted key anchor from verifier bundle",
)
@click.option(
    "--require-attestation/--no-require-attestation",
    default=True,
    show_default=True,
    help="Require attestation.json in bundled PACK.zip (--no-require-attestation is diagnostics only)",
)
@click.option(
    "--validate-profile",
    type=click.Choice(["default", "audit-grade"], case_sensitive=False),
    default="audit-grade",
    show_default=True,
    help="Validation profile for bundled PACK.zip",
)
@click.option(
    "--compare-bundled-validate-report/--no-compare-bundled-validate-report",
    default=True,
    show_default=True,
    help="Compare bundled audit_validate_report.json with recomputed validation result",
)
@click.option(
    "--expect-pack-fingerprint",
    default=None,
    help="Expected PACK fingerprint (checksums.sha256.txt SHA256, 64 hex)",
)
@click.option(
    "--enforce/--no-enforce",
    default=True,
    show_default=True,
    help="Exit non-zero when any issue is found",
)
@click.option(
    "--json-out",
    default=None,
    help="Optional path to write handoff verification result JSON",
)
def audit_handoff_verify(
    handoff_dir: str,
    profile: str,
    strict: bool,
    require_signature: bool,
    require_public_key: bool,
    require_attestation: bool,
    validate_profile: str,
    compare_bundled_validate_report: bool,
    expect_pack_fingerprint: str | None,
    enforce: bool,
    json_out: str | None,
) -> None:
    """Verify handoff folder end-to-end (checksums + audit-verify + attestation + audit-validate)."""
    from autorisk.audit.handoff_verify import verify_audit_handoff

    if str(profile).strip().lower() == "audit-grade":
        click.echo(
            "[audit-grade] "
            + _format_effective_flags(
                {
                    "strict": True,
                    "require_signature": True,
                    "require_public_key": True,
                    "require_attestation": True,
                    "validate_profile": "audit-grade",
                    "compare_bundled_validate_report": True,
                    "trust_embedded_public_key": False,
                    "require_attestation_key_match_signature": True,
                }
            )
        )
    elif str(profile).strip().lower() == "default":
        click.echo(
            "[default] diagnostics mode: does not enforce trusted signature/attestation. Do not use for audit-grade acceptance.",
            err=True,
        )
        click.echo(
            "Use: audit-handoff-verify --profile audit-grade [--expect-pack-fingerprint <TICKET_FP>] --enforce",
            err=True,
        )

    result = verify_audit_handoff(
        handoff_dir,
        profile=profile,
        strict=strict,
        require_signature=require_signature,
        require_public_key=require_public_key,
        require_attestation=require_attestation,
        validate_profile=validate_profile,
        compare_bundled_validate_report=compare_bundled_validate_report,
        expect_pack_fingerprint=expect_pack_fingerprint,
    )

    click.echo(f"Handoff dir: {result.handoff_dir}")
    click.echo(f"Checksums: {result.checksums_path}")
    click.echo(f"Listed files: {result.listed_files}")
    click.echo(f"Verified files: {result.verified_files}")
    click.echo(f"Pack: {result.pack_path}")
    click.echo(f"Verifier bundle: {result.verifier_bundle_zip_path}")
    click.echo(f"Finalize record: {result.finalize_record_path}")
    if result.validate_report_path is not None:
        click.echo(f"Bundled validate report: {result.validate_report_path}")
    click.echo(f"audit-verify ok: {result.audit_verify_ok}")
    click.echo(f"audit-validate ok: {result.audit_validate_ok}")
    click.echo(f"Attestation present: {result.attestation_present}")
    click.echo(f"Attestation verified: {result.attestation_verified}")
    if result.attestation_present:
        click.echo(f"Attestation key id: {result.attestation_key_id}")
        click.echo(f"Attestation key source: {result.attestation_key_source or 'none'}")
    if result.pack_fingerprint != "":
        click.echo(f"Pack fingerprint: {result.pack_fingerprint}")
    if result.expected_pack_fingerprint != "":
        click.echo(f"Expected fingerprint: {result.expected_pack_fingerprint}")
        click.echo(
            f"Expected fingerprint match: {result.expected_pack_fingerprint_match}"
        )
    if result.bundled_validate_report_match is not None:
        click.echo(f"validate report match: {result.bundled_validate_report_match}")
    click.echo(f"Issues: {len(result.issues)}")

    if json_out is not None and str(json_out).strip() != "":
        out_path = Path(json_out).expanduser().resolve()
        out_path.write_text(result.to_json(), encoding="utf-8")
        click.echo(f"Wrote: {out_path}")

    if result.issues:
        for issue in result.issues[:50]:
            click.echo(f"- {issue.kind}: {issue.path} {issue.detail}".rstrip())
        if len(result.issues) > 50:
            click.echo(f"... ({len(result.issues) - 50} more)")
        if enforce:
            raise SystemExit(2)


@click.command("review-approve")
@click.option(
    "--run-dir",
    "-r",
    required=True,
    help="Run directory containing cosmos_results.json",
)
@click.option(
    "--rank", type=int, required=True, help="candidate_rank to approve/override"
)
@click.option(
    "--severity",
    type=click.Choice(["NONE", "LOW", "MEDIUM", "HIGH"], case_sensitive=False),
    required=True,
    help="Final severity after human review",
)
@click.option("--reason", required=True, help="Human reviewer reason (kept for audit)")
@click.option(
    "--evidence", multiple=True, help="Optional evidence references (repeatable)"
)
@click.option(
    "--operator",
    "operator_user",
    default=None,
    help="Override operator username (default: env)",
)
@click.option(
    "--log",
    "log_path",
    default=None,
    help="Optional review log path (default: RUN_DIR/review_log.jsonl)",
)
def review_approve(
    run_dir: str,
    rank: int,
    severity: str,
    reason: str,
    evidence: tuple[str, ...],
    operator_user: str | None,
    log_path: str | None,
) -> None:
    """Append one human review decision into review_log.jsonl."""
    from autorisk.review.log import append_review_decision

    path, record, rec_sha = append_review_decision(
        run_dir=run_dir,
        candidate_rank=rank,
        severity_after=severity,
        reason=reason,
        evidence_refs=list(evidence),
        operator_user=operator_user,
        log_path=log_path,
    )

    click.echo(f"Review log: {path}")
    click.echo(
        f"Recorded: rank={rank} {record['decision_before']['severity']} -> {record['decision_after']['severity']}"
    )
    click.echo(f"Review record sha256: {rec_sha}")


@click.command("review-apply")
@click.option(
    "--run-dir",
    "-r",
    required=True,
    help="Run directory containing cosmos_results.json",
)
@click.option(
    "--log",
    "log_path",
    default=None,
    help="Optional review log path (default: RUN_DIR/review_log.jsonl)",
)
@click.option(
    "--out",
    "output_path",
    default=None,
    help="Output reviewed results path (default: RUN_DIR/cosmos_results_reviewed.json)",
)
@click.option(
    "--allow-stale/--no-allow-stale",
    default=False,
    show_default=True,
    help="Apply reviews even if results_sha256 differs",
)
def review_apply(
    run_dir: str, log_path: str | None, output_path: str | None, allow_stale: bool
) -> None:
    """Apply review_log.jsonl decisions into cosmos_results_reviewed.json (non-destructive)."""
    from autorisk.review.log import apply_review_overrides

    res = apply_review_overrides(
        run_dir=run_dir,
        log_path=log_path,
        output_path=output_path,
        allow_stale=allow_stale,
        write_report=True,
    )

    click.echo(f"Input: {res.input_results}")
    click.echo(f"Review log: {res.log_path}")
    click.echo(f"Output: {res.output_results}")
    click.echo(f"Diff report: {res.diff_report_path}")
    click.echo(f"Applied: {res.applied}")
    click.echo(f"Skipped stale: {res.skipped_stale}")
    click.echo(f"Skipped missing rank: {res.skipped_missing}")


@click.command("policy-check")
@click.option(
    "--run-dir",
    "-r",
    required=True,
    help="Run directory containing cosmos_results.json",
)
@click.option(
    "--policy",
    default=None,
    help="Policy YAML path (default: configs/policy.yaml if present)",
)
@click.option(
    "--review-log",
    default=None,
    help="Optional review log path (default: RUN_DIR/review_log.jsonl)",
)
@click.option(
    "--report-out",
    default=None,
    help="Optional policy report path (default: RUN_DIR/policy_report.json)",
)
@click.option(
    "--queue-out",
    default=None,
    help="Optional review queue path (default: RUN_DIR/review_queue.json)",
)
@click.option(
    "--snapshot-out",
    default=None,
    help="Optional policy snapshot path (default: RUN_DIR/policy_snapshot.json)",
)
@click.option(
    "--allow-stale",
    "allow_stale",
    flag_value=True,
    default=None,
    help="Override policy to allow stale review logs",
)
@click.option(
    "--no-allow-stale",
    "allow_stale",
    flag_value=False,
    help="Override policy to disallow stale review logs",
)
@click.option(
    "--enforce/--no-enforce",
    default=False,
    show_default=True,
    help="Exit non-zero if policy check fails",
)
def policy_check(
    run_dir: str,
    policy: str | None,
    review_log: str | None,
    report_out: str | None,
    queue_out: str | None,
    snapshot_out: str | None,
    allow_stale: bool | None,
    enforce: bool,
) -> None:
    """Enforce review gating policy and emit policy_report/review_queue."""
    from autorisk.policy.check import run_policy_check

    res = run_policy_check(
        run_dir=run_dir,
        policy_path=policy,
        review_log=review_log,
        report_path=report_out,
        queue_path=queue_out,
        snapshot_path=snapshot_out,
        allow_stale=allow_stale,
        write_outputs=True,
    )

    click.echo(f"Policy report: {res.report_path}")
    click.echo(f"Review queue: {res.queue_path}")
    click.echo(f"Policy snapshot: {res.snapshot_path}")
    click.echo(
        f"Policy source: {res.policy_source.get('policy_path') or res.policy_source.get('source_type')}"
    )
    click.echo(f"Passed: {res.passed}")
    click.echo(f"Required review: {res.required_review_count}")
    click.echo(f"Reviewed (valid): {res.reviewed_count_valid}")
    click.echo(f"Reviewed (stale): {res.reviewed_count_stale}")
    click.echo(f"Missing review: {res.missing_review_count}")

    if res.violations:
        for v in res.violations[:50]:
            click.echo(
                f"- rank={v.get('candidate_rank')} sev={v.get('severity')} reasons={','.join(v.get('violation_reasons', []))}"
            )
        if len(res.violations) > 50:
            click.echo(f"... ({len(res.violations) - 50} more)")
        if enforce:
            raise SystemExit(2)


@click.command("finalize-run")
@click.option(
    "--run-dir",
    "-r",
    required=True,
    help="Run directory containing cosmos_results.json",
)
@click.option(
    "--policy",
    default=None,
    help="Policy YAML path (default: configs/policy.yaml if present)",
)
@click.option(
    "--review-log",
    default=None,
    help="Optional review log path (default: RUN_DIR/review_log.jsonl)",
)
@click.option(
    "--input-video",
    "-i",
    default=None,
    help="Optional source input video path (for SHA256 provenance)",
)
@click.option(
    "--out", "-o", "output_dir", default=None, help="Audit pack output directory"
)
@click.option(
    "--include-clips/--no-include-clips",
    default=True,
    show_default=True,
    help="Copy candidate clips into audit pack",
)
@click.option(
    "--zip/--no-zip",
    "create_zip",
    default=True,
    show_default=True,
    help="Create zip bundle for handoff",
)
@click.option(
    "--allow-stale",
    "allow_stale",
    flag_value=True,
    default=None,
    help="Override policy to allow stale review logs",
)
@click.option(
    "--no-allow-stale",
    "allow_stale",
    flag_value=False,
    help="Override policy to disallow stale review logs",
)
@click.option(
    "--enforce/--no-enforce",
    default=True,
    show_default=True,
    help="Fail finalization on policy or integrity violations",
)
@click.option(
    "--audit-grade/--no-audit-grade",
    default=False,
    show_default=True,
    help="Enable strict audit mode (signature/trusted key/enforce)",
)
@click.option(
    "--sign-private-key",
    default=None,
    help="Optional Ed25519 private key PEM for audit-sign",
)
@click.option(
    "--sign-private-key-password",
    default=None,
    help="Private key password (less secure; prefer --sign-private-key-password-env)",
)
@click.option(
    "--sign-private-key-password-env",
    default=None,
    help="Environment variable containing private key password",
)
@click.option(
    "--sign-public-key",
    default=None,
    help="Optional Ed25519 public key PEM for signature verification",
)
@click.option(
    "--sign-public-key-dir",
    default=None,
    help="Optional trusted public key directory for verification",
)
@click.option(
    "--require-signature/--no-require-signature",
    default=False,
    show_default=True,
    help="Require signature.json during final verify",
)
@click.option(
    "--require-trusted-key/--no-require-trusted-key",
    default=False,
    show_default=True,
    help="Require explicit trusted key anchor (--sign-public-key or --sign-public-key-dir)",
)
@click.option(
    "--verifier-bundle-out",
    default=None,
    help="Optional output directory for verifier bundle",
)
@click.option(
    "--write-verifier-bundle/--no-write-verifier-bundle",
    default=None,
    help="Generate verifier bundle (keys/trusted + revoked_key_ids + VERIFY.md)",
)
@click.option(
    "--handoff-out",
    default=None,
    help="Optional output directory for audit handoff folder",
)
@click.option(
    "--write-handoff/--no-write-handoff",
    default=None,
    help="Generate handoff folder (PACK.zip + verifier bundle + finalize record + HANDOFF.md)",
)
@click.option(
    "--embed-public-key/--no-embed-public-key",
    default=False,
    show_default=True,
    help="Embed public key PEM in signature.json when signing",
)
@click.option(
    "--trust-embedded-public-key/--no-trust-embedded-public-key",
    default=False,
    show_default=True,
    help="Allow fallback to embedded public key when --sign-public-key is not provided",
)
@click.option(
    "--revoked-key-id",
    "revoked_key_ids",
    multiple=True,
    help="Revoked signature key_id (repeatable)",
)
@click.option(
    "--revocation-file",
    default=None,
    help="Path to revoked key IDs file (one key_id per line)",
)
@click.pass_context
def finalize_run(
    ctx: click.Context,
    run_dir: str,
    policy: str | None,
    review_log: str | None,
    input_video: str | None,
    output_dir: str | None,
    include_clips: bool,
    create_zip: bool,
    allow_stale: bool | None,
    enforce: bool,
    audit_grade: bool,
    sign_private_key: str | None,
    sign_private_key_password: str | None,
    sign_private_key_password_env: str | None,
    sign_public_key: str | None,
    sign_public_key_dir: str | None,
    require_signature: bool,
    require_trusted_key: bool,
    verifier_bundle_out: str | None,
    write_verifier_bundle: bool | None,
    handoff_out: str | None,
    write_handoff: bool | None,
    embed_public_key: bool,
    trust_embedded_public_key: bool,
    revoked_key_ids: tuple[str, ...],
    revocation_file: str | None,
) -> None:
    """Apply review, enforce policy, build audit pack, and verify integrity."""
    from autorisk.services.audit_service import (
        FinalizeRunServiceRequest,
        run_finalize_run,
    )

    resolved_sign_password = _resolve_private_key_password(
        private_key_password=sign_private_key_password,
        private_key_password_env=sign_private_key_password_env,
    )
    revoked = _load_revoked_key_ids(
        revoked_key_ids=revoked_key_ids,
        revocation_file=revocation_file,
    )

    request = FinalizeRunServiceRequest(
        run_dir=run_dir,
        policy=policy,
        review_log=review_log,
        input_video=input_video,
        output_dir=output_dir,
        include_clips=include_clips,
        create_zip=create_zip,
        allow_stale=allow_stale,
        enforce=enforce,
        audit_grade=audit_grade,
        sign_private_key=sign_private_key,
        sign_public_key=sign_public_key,
        sign_public_key_dir=sign_public_key_dir,
        require_signature=require_signature,
        require_trusted_key=require_trusted_key,
        verifier_bundle_out=verifier_bundle_out,
        write_verifier_bundle=write_verifier_bundle,
        handoff_out=handoff_out,
        write_handoff=write_handoff,
        embed_public_key=embed_public_key,
        trust_embedded_public_key=trust_embedded_public_key,
        revocation_file=revocation_file,
        resolved_sign_password=resolved_sign_password,
        revoked_key_ids=revoked,
        cfg=(ctx.obj or {}).get("cfg"),
    )

    run_finalize_run(
        request,
        emit=lambda message, is_err: click.echo(message, err=is_err),
    )


def register_audit_commands(cli: click.Group) -> None:
    """Register audit/review/policy/finalize related commands."""
    cli.add_command(audit_pack)
    cli.add_command(audit_sign)
    cli.add_command(audit_attest)
    cli.add_command(audit_verify)
    cli.add_command(audit_verifier_bundle)
    cli.add_command(audit_validate)
    cli.add_command(audit_handoff)
    cli.add_command(audit_handoff_verify)
    cli.add_command(review_approve)
    cli.add_command(review_apply)
    cli.add_command(policy_check)
    cli.add_command(finalize_run)
