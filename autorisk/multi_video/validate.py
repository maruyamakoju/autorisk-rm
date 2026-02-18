"""Schema + semantic validation for multi-video artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_RESOURCE_PACKAGE = "autorisk.resources.schemas"
SEVERITIES = {"NONE", "LOW", "MEDIUM", "HIGH"}
SCHEMA_VERSION = 1


@dataclass
class ArtifactValidateIssue:
    kind: str  # parse_error | schema_error | semantic_error | io_error
    detail: str
    path: str = ""


@dataclass
class ArtifactValidateResult:
    artifact_type: str
    source: Path
    issues: list[ArtifactValidateIssue]

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "source": str(self.source),
            "ok": self.ok,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def _load_schema(schema_file: str, *, schema_dir: str | Path | None = None) -> dict[str, Any]:
    if schema_dir is None:
        text = importlib_resources.files(SCHEMA_RESOURCE_PACKAGE).joinpath(schema_file).read_text(encoding="utf-8")
    else:
        path = Path(schema_dir).expanduser().resolve() / schema_file
        text = path.read_text(encoding="utf-8")

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"schema must be JSON object: {schema_file}")
    return payload


def _schema_issues(*, schema: dict[str, Any], instance: Any, path: str) -> list[ArtifactValidateIssue]:
    validator = Draft202012Validator(schema)
    out: list[ArtifactValidateIssue] = []
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        at = ".".join(str(x) for x in err.absolute_path)
        detail = err.message if at == "" else f"{at}: {err.message}"
        out.append(ArtifactValidateIssue(kind="schema_error", detail=detail[:300], path=path))
    return out


def _schema_version_issue(payload: dict[str, Any], *, path: str) -> list[ArtifactValidateIssue]:
    raw = payload.get("schema_version", None)
    try:
        value = int(raw)
    except Exception:
        return [
            ArtifactValidateIssue(
                kind="semantic_error",
                detail=f"schema_version must be integer {SCHEMA_VERSION}, got: {raw!r}",
                path=path,
            )
        ]
    if value != SCHEMA_VERSION:
        return [
            ArtifactValidateIssue(
                kind="semantic_error",
                detail=f"unsupported schema_version={value}; expected {SCHEMA_VERSION}",
                path=path,
            )
        ]
    return []


def _validate_summary_semantics(payload: dict[str, Any]) -> list[ArtifactValidateIssue]:
    issues: list[ArtifactValidateIssue] = []
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        return [ArtifactValidateIssue(kind="semantic_error", detail="sources must be list", path="sources")]

    failed_count = sum(1 for source in sources if not bool((source or {}).get("ok", False)))
    expected_failed = int(payload.get("failed_sources", 0))
    if failed_count != expected_failed:
        issues.append(
            ArtifactValidateIssue(
                kind="semantic_error",
                detail=f"failed_sources mismatch: payload={expected_failed} computed={failed_count}",
                path="failed_sources",
            )
        )

    expected_ok = failed_count == 0
    if bool(payload.get("ok", False)) != expected_ok:
        issues.append(
            ArtifactValidateIssue(
                kind="semantic_error",
                detail=f"ok mismatch: payload={bool(payload.get('ok', False))} computed={expected_ok}",
                path="ok",
            )
        )

    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        steps = source.get("steps", [])
        if not isinstance(steps, list):
            continue
        bad_steps = [i for i, step in enumerate(steps) if not isinstance(step, dict)]
        if bad_steps:
            issues.append(
                ArtifactValidateIssue(
                    kind="semantic_error",
                    detail=f"steps contains non-object rows at indexes: {bad_steps[:10]}",
                    path=f"sources[{idx}].steps",
                )
            )
            continue
        if not bool(source.get("ok", False)):
            failed_steps = [step for step in steps if bool(step.get("ok", True)) is False]
            if len(failed_steps) == 0:
                issues.append(
                    ArtifactValidateIssue(
                        kind="semantic_error",
                        detail="source marked failed but no failed step exists",
                        path=f"sources[{idx}].ok",
                    )
                )
    return issues


def _validate_metrics_semantics(payload: dict[str, Any]) -> list[ArtifactValidateIssue]:
    issues: list[ArtifactValidateIssue] = []
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        return [ArtifactValidateIssue(kind="semantic_error", detail="sources must be list", path="sources")]

    names: list[str] = []
    available_count = 0
    clips_total = 0
    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        name = str(source.get("name", "")).strip()
        if name != "":
            names.append(name)

        available = bool(source.get("available", False))
        if available:
            available_count += 1
        clip_count = int(source.get("clip_count", 0))
        clips_total += clip_count

        parse_count = int(source.get("parse_success_count", 0))
        if parse_count > clip_count:
            issues.append(
                ArtifactValidateIssue(
                    kind="semantic_error",
                    detail=f"parse_success_count ({parse_count}) > clip_count ({clip_count})",
                    path=f"sources[{idx}].parse_success_count",
                )
            )
        expected_rate = (parse_count / clip_count) if clip_count else 0.0
        reported_rate = float(source.get("parse_success_rate", 0.0))
        if abs(reported_rate - expected_rate) > 1e-5:
            issues.append(
                ArtifactValidateIssue(
                    kind="semantic_error",
                    detail=f"parse_success_rate mismatch: payload={reported_rate} computed={expected_rate}",
                    path=f"sources[{idx}].parse_success_rate",
                )
            )

        counts = source.get("severity_counts", {})
        if isinstance(counts, dict):
            unknown = [k for k in counts.keys() if str(k).upper() not in SEVERITIES]
            if unknown:
                issues.append(
                    ArtifactValidateIssue(
                        kind="semantic_error",
                        detail=f"unknown severities in severity_counts: {unknown}",
                        path=f"sources[{idx}].severity_counts",
                    )
                )
            sum_counts = sum(int(v) for v in counts.values())
            if sum_counts != clip_count:
                issues.append(
                    ArtifactValidateIssue(
                        kind="semantic_error",
                        detail=f"severity_counts sum mismatch: payload={sum_counts} clip_count={clip_count}",
                        path=f"sources[{idx}].severity_counts",
                    )
                )

        ttc = source.get("ttc", {})
        if isinstance(ttc, dict):
            ttc_n = int(ttc.get("n_positive_min_ttc", 0))
            spearman_n = int(ttc.get("spearman_n", 0))
            if spearman_n > ttc_n:
                issues.append(
                    ArtifactValidateIssue(
                        kind="semantic_error",
                        detail=f"ttc.spearman_n ({spearman_n}) > ttc.n_positive_min_ttc ({ttc_n})",
                        path=f"sources[{idx}].ttc.spearman_n",
                    )
                )

    if len(set(names)) != len(names):
        issues.append(
            ArtifactValidateIssue(
                kind="semantic_error",
                detail="duplicate source names detected",
                path="sources[].name",
            )
        )

    if int(payload.get("sources_total", 0)) != len(sources):
        issues.append(
            ArtifactValidateIssue(
                kind="semantic_error",
                detail=f"sources_total mismatch: payload={int(payload.get('sources_total', 0))} computed={len(sources)}",
                path="sources_total",
            )
        )
    if int(payload.get("sources_available", 0)) != available_count:
        issues.append(
            ArtifactValidateIssue(
                kind="semantic_error",
                detail=f"sources_available mismatch: payload={int(payload.get('sources_available', 0))} computed={available_count}",
                path="sources_available",
            )
        )
    if int(payload.get("clips_total", 0)) != clips_total:
        issues.append(
            ArtifactValidateIssue(
                kind="semantic_error",
                detail=f"clips_total mismatch: payload={int(payload.get('clips_total', 0))} computed={clips_total}",
                path="clips_total",
            )
        )
    return issues


def validate_multi_video_run_summary(
    path: str | Path,
    *,
    schema_dir: str | Path | None = None,
) -> ArtifactValidateResult:
    source = Path(path).expanduser().resolve()
    issues: list[ArtifactValidateIssue] = []
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        return ArtifactValidateResult(
            artifact_type="run_summary",
            source=source,
            issues=[ArtifactValidateIssue(kind="parse_error", detail=str(exc)[:300], path=str(source))],
        )
    if not isinstance(payload, dict):
        return ArtifactValidateResult(
            artifact_type="run_summary",
            source=source,
            issues=[ArtifactValidateIssue(kind="parse_error", detail="payload must be object", path=str(source))],
        )

    schema = _load_schema("run_summary.schema.json", schema_dir=schema_dir)
    issues.extend(_schema_issues(schema=schema, instance=payload, path=str(source)))
    issues.extend(_schema_version_issue(payload, path=str(source)))
    issues.extend(_validate_summary_semantics(payload))
    return ArtifactValidateResult(artifact_type="run_summary", source=source, issues=issues)


def validate_submission_metrics(
    path: str | Path,
    *,
    schema_dir: str | Path | None = None,
) -> ArtifactValidateResult:
    source = Path(path).expanduser().resolve()
    issues: list[ArtifactValidateIssue] = []
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        return ArtifactValidateResult(
            artifact_type="submission_metrics",
            source=source,
            issues=[ArtifactValidateIssue(kind="parse_error", detail=str(exc)[:300], path=str(source))],
        )
    if not isinstance(payload, dict):
        return ArtifactValidateResult(
            artifact_type="submission_metrics",
            source=source,
            issues=[ArtifactValidateIssue(kind="parse_error", detail="payload must be object", path=str(source))],
        )

    schema = _load_schema("submission_metrics.schema.json", schema_dir=schema_dir)
    issues.extend(_schema_issues(schema=schema, instance=payload, path=str(source)))
    issues.extend(_schema_version_issue(payload, path=str(source)))
    issues.extend(_validate_metrics_semantics(payload))
    return ArtifactValidateResult(artifact_type="submission_metrics", source=source, issues=issues)
