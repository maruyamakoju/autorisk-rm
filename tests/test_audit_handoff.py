from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from omegaconf import OmegaConf

from autorisk.audit.handoff import build_audit_handoff
from autorisk.audit.pack import build_audit_pack
from autorisk.audit.verifier_bundle import build_verifier_bundle
from autorisk.cli import cli


def _sample_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "cosmos": {
                "backend": "local",
                "local": {
                    "model_name": "nvidia/Cosmos-Reason2-8B",
                    "max_new_tokens": 64,
                    "temperature": 0.2,
                    "torch_dtype": "float16",
                },
                "local_fps": 4,
            }
        }
    )


def _write_fake_trusted_key(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "-----BEGIN PUBLIC KEY-----",
                "dGVzdC1hdWRpdC12ZXJpZmllci1rZXk=",
                "-----END PUBLIC KEY-----",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parse_checksum_rows(path: Path) -> list[str]:
    rows: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "":
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            continue
        rows.append(parts[1])
    return rows


def test_build_audit_handoff_creates_single_submission_bundle(sample_run_dir: Path, tmp_path: Path) -> None:
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    key_src_dir = tmp_path / "trusted_src"
    _write_fake_trusted_key(key_src_dir / "active.pem")
    verifier_res = build_verifier_bundle(
        output_dir=sample_run_dir / "verifier_bundle",
        public_key_dir=key_src_dir,
        revoked_key_ids=set(),
        verify_pack_reference=pack_res.zip_path.name,
    )

    finalize_record_path = sample_run_dir / "finalize_record.json"
    finalize_record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pack_fingerprint": pack_res.checksums_sha256,
                "signature_present": False,
                "signature_verified": None,
                "signature_key_id": "",
                "policy_source": {},
                "policy_sha256": "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sample_run_dir / "audit_validate_report.json").write_text(
        json.dumps(
            {
                "source": str(pack_res.output_dir),
                "mode": "dir",
                "pack_root": str(pack_res.output_dir),
                "schema_dir": "package://autorisk.resources.schemas",
                "files_validated": 1,
                "records_validated": 1,
                "ok": True,
                "issues": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    res = build_audit_handoff(
        run_dir=sample_run_dir,
        output_dir=tmp_path / "handoff",
        pack_zip=pack_res.zip_path,
        verifier_bundle_dir=verifier_res.output_dir,
        finalize_record=finalize_record_path,
    )
    assert res.output_dir.exists()
    assert res.pack_zip_path.exists()
    assert res.verifier_bundle_zip_path.exists()
    assert res.finalize_record_path.exists()
    assert res.validate_report_path is not None and res.validate_report_path.exists()
    assert res.handoff_guide_path.exists()
    assert res.checksums_path.exists()
    guide_text = res.handoff_guide_path.read_text(encoding="utf-8")
    assert "audit-verify -p PACK.zip --profile audit-grade" in guide_text
    assert "--require-signature" not in guide_text

    rows = set(_parse_checksum_rows(res.checksums_path))
    # PACK.zip is verified via audit-verify (signature+attestation), not via handoff checksums.
    assert rows == {"verifier_bundle.zip", "HANDOFF.md", "audit_validate_report.json"}


def test_audit_handoff_cli(sample_run_dir: Path, tmp_path: Path) -> None:
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    key_src_dir = tmp_path / "trusted_src_cli"
    _write_fake_trusted_key(key_src_dir / "active.pem")
    verifier_res = build_verifier_bundle(
        output_dir=sample_run_dir / "verifier_bundle",
        public_key_dir=key_src_dir,
        revoked_key_ids=set(),
        verify_pack_reference=pack_res.zip_path.name,
    )
    finalize_record_path = sample_run_dir / "finalize_record.json"
    finalize_record_path.write_text('{"schema_version":1}', encoding="utf-8")

    runner = CliRunner()
    out_dir = tmp_path / "handoff_cli"
    cli_res = runner.invoke(
        cli,
        [
            "audit-handoff",
            "-r",
            str(sample_run_dir),
            "--out",
            str(out_dir),
            "--pack-zip",
            str(pack_res.zip_path),
            "--verifier-bundle-dir",
            str(verifier_res.output_dir),
            "--finalize-record",
            str(finalize_record_path),
        ],
        catch_exceptions=False,
    )
    assert cli_res.exit_code == 0, cli_res.output
    assert "Handoff directory:" in cli_res.output
    assert (out_dir / "PACK.zip").exists()
    assert (out_dir / "verifier_bundle.zip").exists()
    assert (out_dir / "finalize_record.json").exists()
    assert (out_dir / "HANDOFF.md").exists()
    assert (out_dir / "handoff_checksums.sha256.txt").exists()
