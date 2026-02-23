"""Service-layer orchestration for audit/finalize flows."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autorisk.audit.contracts import FinalizeRunRecord, write_finalize_record

if TYPE_CHECKING:
    from autorisk.multi_video.validate import ArtifactValidateResult

EmitFn = Callable[[str, bool], None]


@dataclass(slots=True)
class FinalizeRunServiceRequest:
    run_dir: str
    policy: str | None
    review_log: str | None
    input_video: str | None
    output_dir: str | None
    include_clips: bool
    create_zip: bool
    allow_stale: bool | None
    enforce: bool
    audit_grade: bool
    sign_private_key: str | None
    sign_public_key: str | None
    sign_public_key_dir: str | None
    require_signature: bool
    require_trusted_key: bool
    verifier_bundle_out: str | None
    write_verifier_bundle: bool | None
    handoff_out: str | None
    write_handoff: bool | None
    embed_public_key: bool
    trust_embedded_public_key: bool
    revocation_file: str | None
    resolved_sign_password: str | None
    revoked_key_ids: set[str]
    cfg: Any | None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _autorisk_version() -> str:
    try:
        return str(importlib.metadata.version("autorisk-rm"))
    except Exception:
        try:
            from autorisk import __version__

            return str(__version__)
        except Exception:
            return "unknown"


def _build_finalize_run_contract_artifacts(
    *,
    run_dir: Path,
    policy_source: dict[str, object],
) -> tuple[Path, "ArtifactValidateResult", Path, "ArtifactValidateResult"]:
    """Generate and validate multi-video contract artifacts for finalize-run."""
    from autorisk.multi_video.submission_metrics import (
        source_submission_summary,
        write_submission_metrics,
    )
    from autorisk.multi_video.validate import (
        validate_multi_video_run_summary,
        validate_submission_metrics,
    )

    run_dir_path = run_dir.resolve()
    clips_dir = run_dir_path / "clips"
    results_path = run_dir_path / "cosmos_results.json"
    run_name = run_dir_path.name or "run"

    clips_total = (
        len(list(clips_dir.glob("candidate_*.mp4"))) if clips_dir.exists() else 0
    )
    results_done = 0
    if results_path.exists():
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                results_done = len(payload)
        except Exception:
            results_done = 0

    now_iso = _utc_now_iso()
    config_hint = str(policy_source.get("policy_path", "") or "configs/policy.yaml")
    run_summary_payload: dict[str, object] = {
        "schema_version": 1,
        "started_at_utc": now_iso,
        "finished_at_utc": now_iso,
        "elapsed_sec": 0.0,
        "dry_run": True,
        "resume": True,
        "fail_fast": False,
        "skip": {
            "supplement": True,
            "ttc": True,
            "grounding": True,
            "report": True,
        },
        "sources": [
            {
                "name": run_name,
                "config_path": config_hint,
                "output_dir": str(run_dir_path),
                "default_video_path": "",
                "started_at_utc": now_iso,
                "finished_at_utc": now_iso,
                "elapsed_sec": 0.0,
                "ok": True,
                "clips_total": int(clips_total),
                "results_done_before": int(results_done),
                "results_done_after_infer": int(results_done),
                "clips_total_after_infer": int(clips_total),
                "steps": [
                    {
                        "label": "finalize-run/contracts",
                        "ok": True,
                        "skipped": True,
                        "reason": "generated during finalize-run",
                        "returncode": 0,
                        "elapsed_sec": 0.0,
                    }
                ],
            }
        ],
        "ok": True,
        "failed_sources": 0,
    }
    run_summary_path = run_dir_path / "run_summary.json"
    _write_json(run_summary_path, run_summary_payload)
    run_summary_validate_res = validate_multi_video_run_summary(run_summary_path)

    source_summary = source_submission_summary(
        repo_root=run_dir_path.parent,
        source={
            "name": run_name,
            "config_path": config_hint,
            "output_dir": str(run_dir_path),
        },
    )
    submission_payload = {
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "sources_total": 1,
        "sources_available": 1 if bool(source_summary.get("available", False)) else 0,
        "clips_total": int(source_summary.get("clip_count", 0)),
        "sources": [source_summary],
    }
    submission_metrics_path = write_submission_metrics(
        submission_payload,
        output_path=run_dir_path / "submission_metrics.json",
    )
    submission_metrics_validate_res = validate_submission_metrics(
        submission_metrics_path
    )
    return (
        run_summary_path,
        run_summary_validate_res,
        submission_metrics_path,
        submission_metrics_validate_res,
    )


def _upsert_zip_member(zip_path: Path, member_name: str, payload: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="autorisk-finalize-") as tmp_dir:
        tmp_zip = Path(tmp_dir) / zip_path.name
        with (
            zipfile.ZipFile(zip_path, "r") as src,
            zipfile.ZipFile(
                tmp_zip,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as dst,
        ):
            for info in src.infolist():
                if info.filename == member_name:
                    continue
                if info.is_dir():
                    dst.writestr(info, b"")
                    continue
                dst.writestr(info, src.read(info.filename))
            dst.writestr(member_name, payload)
        shutil.copy2(tmp_zip, zip_path)


def run_finalize_run(
    request: FinalizeRunServiceRequest,
    *,
    emit: EmitFn,
) -> None:
    """Execute finalize-run orchestration without Click-bound logic."""
    from autorisk.audit.attestation import attest_audit_pack
    from autorisk.audit.handoff import build_audit_handoff
    from autorisk.audit.pack import build_audit_pack
    from autorisk.audit.sign import sign_audit_pack
    from autorisk.audit.validate import validate_audit_pack
    from autorisk.audit.verifier_bundle import build_verifier_bundle
    from autorisk.audit.verify import verify_audit_pack
    from autorisk.policy.check import resolve_policy, run_policy_check
    from autorisk.review.log import apply_review_overrides

    enforce = bool(request.enforce)
    require_signature = bool(request.require_signature)
    require_trusted_key = bool(request.require_trusted_key)
    trust_embedded_public_key = bool(request.trust_embedded_public_key)
    revoked = set(request.revoked_key_ids)

    if request.audit_grade:
        enforce = True
        require_signature = True
        require_trusted_key = True
        trust_embedded_public_key = False
        emit(
            "[audit-grade] enforce=true require_signature=true require_trusted_key=true trust_embedded_public_key=false",
            False,
        )

    if request.audit_grade and (
        request.sign_private_key is None or str(request.sign_private_key).strip() == ""
    ):
        emit("Error: --audit-grade requires --sign-private-key", True)
        raise SystemExit(2)
    if request.audit_grade:
        has_sign_public_key = (
            request.sign_public_key is not None
            and str(request.sign_public_key).strip() != ""
        )
        has_sign_public_key_dir = (
            request.sign_public_key_dir is not None
            and str(request.sign_public_key_dir).strip() != ""
        )
        if not has_sign_public_key and not has_sign_public_key_dir:
            emit(
                "Error: --audit-grade requires --sign-public-key or --sign-public-key-dir",
                True,
            )
            raise SystemExit(2)

    write_verifier_bundle_effective = (
        bool(request.write_verifier_bundle)
        if request.write_verifier_bundle is not None
        else bool(request.audit_grade)
    )
    write_handoff_effective = (
        bool(request.write_handoff)
        if request.write_handoff is not None
        else bool(request.audit_grade)
    )
    if write_handoff_effective:
        write_verifier_bundle_effective = True
    if write_handoff_effective and not request.create_zip:
        emit("Error: --write-handoff requires --zip (handoff requires PACK.zip)", True)
        raise SystemExit(2)

    effective_policy, policy_source = resolve_policy(
        policy_path=request.policy,
        required_review_severities=None,
        require_parse_failure_review=None,
        require_error_review=None,
        allow_stale=request.allow_stale,
    )
    allow_stale_effective = bool(effective_policy["allow_stale"])
    emit(
        f"[0/4] policy: source={policy_source.get('policy_path') or policy_source.get('source_type')} allow_stale={allow_stale_effective}",
        False,
    )

    apply_res = apply_review_overrides(
        run_dir=request.run_dir,
        log_path=request.review_log,
        output_path=None,
        allow_stale=allow_stale_effective,
        write_report=True,
    )
    emit(f"[1/4] review-apply: {apply_res.output_results}", False)
    emit(
        f"        applied={apply_res.applied} stale={apply_res.skipped_stale} missing={apply_res.skipped_missing}",
        False,
    )

    policy_res = run_policy_check(
        run_dir=request.run_dir,
        policy_path=request.policy,
        review_log=request.review_log,
        allow_stale=allow_stale_effective,
        write_outputs=True,
    )
    emit(
        f"[2/4] policy-check: passed={policy_res.passed} missing={policy_res.missing_review_count}",
        False,
    )
    emit(f"        report={policy_res.report_path}", False)
    emit(f"        queue={policy_res.queue_path}", False)
    if enforce and not policy_res.passed:
        raise SystemExit(2)

    if request.audit_grade:
        (
            run_summary_path,
            run_summary_validate_res,
            submission_metrics_path,
            submission_metrics_validate_res,
        ) = _build_finalize_run_contract_artifacts(
            run_dir=Path(request.run_dir),
            policy_source=policy_source,
        )
        run_summary_issues = len(run_summary_validate_res.issues)
        submission_metrics_issues = len(submission_metrics_validate_res.issues)
        emit(
            f"[2.5/4] multi-validate: run_summary_issues={run_summary_issues} submission_metrics_issues={submission_metrics_issues}",
            False,
        )
        emit(f"        run_summary={run_summary_path}", False)
        emit(f"        submission_metrics={submission_metrics_path}", False)
        for issue in run_summary_validate_res.issues[:20]:
            emit(f"- run_summary {issue.kind}: {issue.detail}".rstrip(), True)
        for issue in submission_metrics_validate_res.issues[:20]:
            emit(f"- submission_metrics {issue.kind}: {issue.detail}".rstrip(), True)
        if run_summary_issues > 0 or submission_metrics_issues > 0:
            raise SystemExit(2)

    pack_res = build_audit_pack(
        run_dir=request.run_dir,
        cfg=request.cfg,
        output_dir=request.output_dir,
        input_video=request.input_video,
        review_log=request.review_log,
        include_clips=request.include_clips,
        create_zip=request.create_zip,
    )
    emit(f"[3/4] audit-pack: {pack_res.output_dir}", False)
    emit(f"        fingerprint={pack_res.checksums_sha256}", False)
    if pack_res.zip_path is not None:
        emit(f"        zip={pack_res.zip_path}", False)

    verify_target = (
        pack_res.zip_path if pack_res.zip_path is not None else pack_res.output_dir
    )
    if (
        request.sign_private_key is not None
        and str(request.sign_private_key).strip() != ""
    ):
        has_sign_public_key = (
            request.sign_public_key is not None
            and str(request.sign_public_key).strip() != ""
        )
        has_sign_public_key_dir = (
            request.sign_public_key_dir is not None
            and str(request.sign_public_key_dir).strip() != ""
        )
        if not has_sign_public_key and not has_sign_public_key_dir:
            warning_message = (
                "--sign-private-key was provided without --sign-public-key/--sign-public-key-dir. "
                "Authenticity should be anchored to an external trusted key."
            )
            if request.audit_grade or require_trusted_key:
                emit(f"Error: {warning_message}", True)
                raise SystemExit(2)
            emit(f"Warning: {warning_message}", True)
        sign_res = sign_audit_pack(
            verify_target,
            private_key_path=request.sign_private_key,
            private_key_password=request.resolved_sign_password,
            public_key_path=request.sign_public_key,
            include_public_key=request.embed_public_key,
        )
        emit(f"[3.5/4] audit-sign: {sign_res.signature_path}", False)
        emit(f"        key_id={sign_res.key_id}", False)

    has_trusted_anchor = (
        request.sign_public_key is not None
        and str(request.sign_public_key).strip() != ""
    ) or (
        request.sign_public_key_dir is not None
        and str(request.sign_public_key_dir).strip() != ""
    )
    if require_trusted_key and not has_trusted_anchor:
        emit(
            "Error: --require-trusted-key requires --sign-public-key or --sign-public-key-dir",
            True,
        )
        raise SystemExit(2)

    verify_res = verify_audit_pack(
        verify_target,
        strict=True,
        public_key=request.sign_public_key,
        public_key_dir=request.sign_public_key_dir,
        require_signature=require_signature,
        require_public_key=require_trusted_key,
        trust_embedded_public_key=trust_embedded_public_key,
        revoked_key_ids=revoked,
    )
    emit(
        f"[4/4] audit-verify: issues={len(verify_res.issues)} expected={verify_res.expected_files} verified={verify_res.verified_files}",
        False,
    )
    if verify_res.signature_present:
        emit(
            f"        signature_verified={verify_res.signature_verified} key_id={verify_res.signature_key_id}",
            False,
        )

    verifier_bundle_path = ""
    if write_verifier_bundle_effective:
        bundle_out = (
            Path(request.verifier_bundle_out).expanduser().resolve()
            if request.verifier_bundle_out is not None
            and str(request.verifier_bundle_out).strip() != ""
            else (Path(request.run_dir).resolve() / "verifier_bundle")
        )
        bundle_res = build_verifier_bundle(
            output_dir=bundle_out,
            public_key=request.sign_public_key,
            public_key_dir=request.sign_public_key_dir,
            revoked_key_ids=revoked,
            revocation_file=request.revocation_file,
            verify_pack_reference=(
                pack_res.zip_path.name
                if pack_res.zip_path is not None
                else pack_res.output_dir.name
            ),
        )
        verifier_bundle_path = str(bundle_res.output_dir)
        emit(f"[4.5/4] verifier-bundle: {bundle_res.output_dir}", False)

    run_dir_path = Path(request.run_dir).resolve()
    validate_report_path = run_dir_path / "audit_validate_report.json"
    pack_finalize_record_path = (
        pack_res.output_dir / "run_artifacts" / "finalize_record.json"
    )
    pack_validate_report_path = (
        pack_res.output_dir / "run_artifacts" / "audit_validate_report.json"
    )

    validate_res = validate_audit_pack(
        verify_target,
        semantic_checks=True,
        profile="default",
    )
    validate_report_payload = validate_res.to_json()
    validate_report_path.write_text(validate_report_payload, encoding="utf-8")
    validate_report_sha256 = _sha256_file(validate_report_path)

    handoff_path = ""
    handoff_checksums_sha256 = ""
    handoff_pack_zip_sha256 = ""
    handoff_verifier_bundle_zip_sha256 = ""
    revocation_file_sha256 = ""
    revocation_path_resolved = ""
    if (
        request.revocation_file is not None
        and str(request.revocation_file).strip() != ""
    ):
        rev_path = Path(request.revocation_file).expanduser().resolve()
        if rev_path.exists() and rev_path.is_file():
            revocation_path_resolved = str(rev_path)
            revocation_file_sha256 = _sha256_file(rev_path)

    finalize_record = FinalizeRunRecord.model_validate(
        {
            "schema_version": 1,
            "created_at_utc": _utc_now_iso(),
            "run_dir": str(run_dir_path),
            "pack_dir": str(pack_res.output_dir),
            "zip_path": str(pack_res.zip_path) if pack_res.zip_path is not None else "",
            "autorisk_version": _autorisk_version(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "pack_fingerprint": pack_res.checksums_sha256,
            "signature_present": bool(verify_res.signature_present),
            "signature_verified": verify_res.signature_verified,
            "signature_key_id": verify_res.signature_key_id,
            "signature_key_source": verify_res.signature_key_source,
            "policy_source": policy_res.policy_source,
            "policy_sha256": str(policy_res.policy_source.get("policy_sha256", "")),
            "revocation_file": revocation_path_resolved,
            "revocation_file_sha256": revocation_file_sha256,
            "revoked_key_ids_count": len(revoked),
            "verification_issues": len(verify_res.issues),
            "enforce": bool(enforce),
            "audit_grade": bool(request.audit_grade),
            "require_signature": bool(require_signature),
            "require_trusted_key": bool(require_trusted_key),
            "trust_embedded_public_key": bool(trust_embedded_public_key),
            "verifier_bundle_path": verifier_bundle_path,
            "validate_ok": bool(validate_res.ok),
            "validate_issues_count": len(validate_res.issues),
            "validate_report_path": "run_artifacts/audit_validate_report.json",
            "validate_report_sha256": validate_report_sha256,
            "handoff_path": handoff_path,
            "handoff_checksums_sha256": handoff_checksums_sha256,
            "handoff_pack_zip_sha256": handoff_pack_zip_sha256,
            "handoff_verifier_bundle_zip_sha256": handoff_verifier_bundle_zip_sha256,
        }
    )
    finalize_record_path = run_dir_path / "finalize_record.json"

    def _sync_run_finalize(
        record: FinalizeRunRecord | dict[str, object],
    ) -> FinalizeRunRecord:
        return write_finalize_record(finalize_record_path, record)

    def _sync_pack_metadata(
        record: FinalizeRunRecord | dict[str, object], report_payload: str
    ) -> FinalizeRunRecord:
        record_obj = write_finalize_record(pack_finalize_record_path, record)
        pack_validate_report_path.write_text(report_payload, encoding="utf-8")
        if pack_res.zip_path is not None:
            _upsert_zip_member(
                pack_res.zip_path,
                "run_artifacts/finalize_record.json",
                record_obj.to_json_text().encode("utf-8"),
            )
            _upsert_zip_member(
                pack_res.zip_path,
                "run_artifacts/audit_validate_report.json",
                report_payload.encode("utf-8"),
            )
        return record_obj

    finalize_record = _sync_run_finalize(finalize_record)
    _sync_pack_metadata(finalize_record, validate_report_payload)

    validate_profile = "audit-grade" if request.audit_grade else "default"
    validate_res = validate_audit_pack(
        verify_target,
        semantic_checks=True,
        profile=validate_profile,
    )
    validate_report_payload = validate_res.to_json()
    validate_report_path.write_text(validate_report_payload, encoding="utf-8")
    validate_report_sha256 = _sha256_file(validate_report_path)
    finalize_record.validate_ok = bool(validate_res.ok)
    finalize_record.validate_issues_count = len(validate_res.issues)
    finalize_record.validate_report_sha256 = validate_report_sha256
    finalize_record = _sync_run_finalize(finalize_record)
    _sync_pack_metadata(finalize_record, validate_report_payload)
    emit(
        f"[4.6/4] audit-validate: issues={len(validate_res.issues)} profile={validate_profile}",
        False,
    )

    handoff_res = None
    if write_handoff_effective:
        if pack_res.zip_path is None:
            emit("Error: handoff requires a zip pack, but zip is missing", True)
            raise SystemExit(2)
        if verifier_bundle_path == "":
            emit(
                "Error: handoff requires verifier bundle, but it was not generated",
                True,
            )
            raise SystemExit(2)
        handoff_output_dir = (
            Path(request.handoff_out).expanduser().resolve()
            if request.handoff_out is not None
            and str(request.handoff_out).strip() != ""
            else (run_dir_path / "handoff_latest")
        )
        handoff_res = build_audit_handoff(
            run_dir=run_dir_path,
            output_dir=handoff_output_dir,
            pack_zip=pack_res.zip_path,
            verifier_bundle_dir=Path(verifier_bundle_path),
            finalize_record=finalize_record_path,
        )
        handoff_path = str(handoff_res.output_dir)
        handoff_checksums_sha256 = _sha256_file(handoff_res.checksums_path)
        handoff_verifier_bundle_zip_sha256 = _sha256_file(
            handoff_res.verifier_bundle_zip_path
        )
        finalize_record.handoff_path = handoff_path
        finalize_record.handoff_checksums_sha256 = handoff_checksums_sha256
        finalize_record.handoff_verifier_bundle_zip_sha256 = (
            handoff_verifier_bundle_zip_sha256
        )
        finalize_record = _sync_run_finalize(finalize_record)
        pack_finalize_record = finalize_record.model_copy(deep=True)
        pack_finalize_record.handoff_path = ""
        pack_finalize_record.handoff_checksums_sha256 = ""
        pack_finalize_record.handoff_pack_zip_sha256 = ""
        pack_finalize_record.handoff_verifier_bundle_zip_sha256 = ""
        pack_finalize_record.handoff_anchor_checksums_sha256 = handoff_checksums_sha256
        pack_finalize_record.handoff_anchor_verifier_bundle_zip_sha256 = (
            handoff_verifier_bundle_zip_sha256
        )
        _sync_pack_metadata(pack_finalize_record, validate_report_payload)
    else:
        pack_finalize_record = finalize_record.model_copy(deep=True)

    if (
        request.sign_private_key is not None
        and str(request.sign_private_key).strip() != ""
    ):
        attest_res = attest_audit_pack(
            verify_target,
            private_key_path=request.sign_private_key,
            private_key_password=request.resolved_sign_password,
            public_key_path=request.sign_public_key,
            include_public_key=request.embed_public_key,
        )
        emit(f"[4.65/4] audit-attest: {attest_res.attestation_path}", False)
        emit(f"        key_id={attest_res.key_id}", False)

    final_verify_res = verify_res
    if request.audit_grade:
        final_verify_res = verify_audit_pack(
            verify_target,
            strict=True,
            public_key=request.sign_public_key,
            public_key_dir=request.sign_public_key_dir,
            require_signature=True,
            require_public_key=True,
            require_attestation=True,
            require_attestation_key_match_signature=True,
            trust_embedded_public_key=False,
            revoked_key_ids=revoked,
        )
        emit(
            f"[4.66/4] audit-verify(final): issues={len(final_verify_res.issues)} signature={final_verify_res.signature_verified} attestation={final_verify_res.attestation_verified}",
            False,
        )

    if enforce and (
        verify_res.issues or validate_res.issues or final_verify_res.issues
    ):
        raise SystemExit(2)

    if handoff_res is not None and pack_res.zip_path is not None:
        shutil.copy2(pack_res.zip_path, handoff_res.pack_zip_path)
        handoff_pack_zip_sha256 = _sha256_file(handoff_res.pack_zip_path)
        finalize_record.handoff_pack_zip_sha256 = handoff_pack_zip_sha256
        finalize_record = _sync_run_finalize(finalize_record)
        handoff_finalize_record_path = Path(handoff_path) / "finalize_record.json"
        write_finalize_record(handoff_finalize_record_path, finalize_record)
        emit(f"[4.7/4] audit-handoff: {handoff_res.output_dir}", False)

    emit(f"[4.8/4] finalize-record: {finalize_record_path}", False)
