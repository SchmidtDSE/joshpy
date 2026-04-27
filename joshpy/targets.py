"""Target profile system for batch remote execution.

Reads and writes ``~/.josh/targets/<name>.json`` — the shared config format
between josh (Java) and joshpy (Python).  Each profile defines a deployment
target (HTTP Cloud Run or Kubernetes) with connection info, MinIO/S3
credentials, and resource settings.

Example usage::

    from joshpy.targets import (
        TargetProfile, HttpTargetConfig, KubernetesTargetConfig,
        save_target, load_target, list_targets, resolve_minio_creds,
    )

    # Create and save an HTTP target
    profile = TargetProfile(
        target_type="http",
        http=HttpTargetConfig(endpoint="https://josh.example.com"),
        minio_endpoint="https://storage.googleapis.com",
        minio_bucket="josh-results",
    )
    save_target("cloud-dev", profile)

    # Load and resolve credentials
    loaded = load_target("cloud-dev")
    creds = resolve_minio_creds(loaded)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGETS_DIR: Path = Path.home() / ".josh" / "targets"

_VALID_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")

TargetType = Literal["http", "kubernetes"]

# Python field name -> JSON key  (only where they differ)
_TO_JSON: dict[str, str] = {
    "target_type": "type",
    "api_key": "apiKey",
    "timeout_seconds": "timeoutSeconds",
    "ttl_seconds_after_finished": "ttlSecondsAfterFinished",
    "node_selector": "nodeSelector",
}
_FROM_JSON: dict[str, str] = {v: k for k, v in _TO_JSON.items()}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpTargetConfig:
    """HTTP target configuration (Cloud Run / standalone server).

    Attributes:
        endpoint: Server URL (required).
        api_key: Optional API key for authentication.
    """

    endpoint: str
    api_key: str | None = None


@dataclass(frozen=True)
class KubernetesTargetConfig:
    """Kubernetes target configuration (GKE / any k8s cluster).

    Attributes:
        namespace: Kubernetes namespace for jobs (required).
        image: Container image for simulation pods (required).
        context: kubectl context name (None = current context).
        pod_minio_endpoint: In-cluster MinIO endpoint pods use (may differ
            from the host-side ``minio_endpoint`` on TargetProfile).
        resources: K8s resource spec, e.g.
            ``{"requests": {"cpu": "1", "memory": "2Gi"},
              "limits": {"memory": "4Gi"}}``.
        parallelism: Max concurrent pods per job.
        timeout_seconds: Job timeout in seconds.
        ttl_seconds_after_finished: Auto-cleanup delay after job completes.
        spot: Use preemptible / spot nodes.
        node_selector: K8s pod-spec ``nodeSelector`` map (label key -> value)
            that constrains pods to nodes carrying matching labels. On GKE
            Autopilot, set ``{"cloud.google.com/compute-class": "Balanced"}``
            to access the Balanced compute class (up to 222 vCPU / 851 GiB
            per pod vs the default class's 30 vCPU / 110 GiB cap). Layers
            additively with ``spot``: setting both yields a Spot Balanced
            pod (~60-90% cost savings on a high-memory class).
    """

    namespace: str
    image: str
    context: str | None = None
    pod_minio_endpoint: str | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    parallelism: int | None = None
    timeout_seconds: int | None = None
    ttl_seconds_after_finished: int | None = None
    spot: bool = False
    node_selector: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetProfile:
    """Top-level target profile container.

    Attributes:
        target_type: ``"http"`` or ``"kubernetes"``.
        http: HTTP config (required when target_type is ``"http"``).
        kubernetes: K8s config (required when target_type is ``"kubernetes"``).
        minio_endpoint: Host-side MinIO/S3 endpoint for staging.
        minio_access_key: MinIO access key (prefer env vars for secrets).
        minio_secret_key: MinIO secret key (prefer env vars for secrets).
        minio_bucket: MinIO/GCS bucket name.
    """

    target_type: TargetType
    http: HttpTargetConfig | None = None
    kubernetes: KubernetesTargetConfig | None = None
    minio_endpoint: str | None = None
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str | None = None

    def __post_init__(self) -> None:
        if self.target_type == "http" and self.http is None:
            raise ValueError(
                "http config required when target_type='http'"
            )
        if self.target_type == "kubernetes" and self.kubernetes is None:
            raise ValueError(
                "kubernetes config required when target_type='kubernetes'"
            )


@dataclass(frozen=True)
class ResolvedMinioCreds:
    """Fully resolved MinIO credentials (profile + env vars merged).

    Attributes:
        endpoint: MinIO/S3 endpoint URL.
        access_key: Access key.
        secret_key: Secret key.
        bucket: Bucket name.
    """

    endpoint: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    bucket: str | None = None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_json_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a JSON-compatible dict.

    Renames fields via ``_TO_JSON``, omits ``None`` values, and preserves
    nested dicts (like ``resources``) as-is.
    """
    result: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if value is None:
            continue
        # Skip false booleans only when they match the field default
        if isinstance(value, bool) and not value and f.default is False:
            continue
        # Skip empty dicts when that is the default
        if isinstance(value, dict) and not value:
            continue
        key = _TO_JSON.get(f.name, f.name)
        result[key] = value
    return result


def _from_json_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Rename JSON keys to Python field names via ``_FROM_JSON``."""
    return {_FROM_JSON.get(k, k): v for k, v in d.items()}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _profile_path(name: str) -> Path:
    """Validate *name* and return ``TARGETS_DIR / f"{name}.json"``.

    Raises:
        ValueError: If *name* contains characters outside ``[a-zA-Z0-9_-]``.
    """
    if not _VALID_NAME.match(name):
        raise ValueError(
            f"Invalid target name {name!r}: "
            "must match [a-zA-Z0-9_-]+ (no dots, slashes, or spaces)."
        )
    return TARGETS_DIR / f"{name}.json"


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def save_target(name: str, profile: TargetProfile) -> Path:
    """Write a target profile to ``~/.josh/targets/<name>.json``.

    Creates the directory if it does not exist.

    Args:
        name: Profile name (alphanumeric, hyphens, underscores).
        profile: Target profile to save.

    Returns:
        Path to the written JSON file.
    """
    path = _profile_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {"type": profile.target_type}

    if profile.http is not None:
        data["http"] = _to_json_dict(profile.http)
    if profile.kubernetes is not None:
        data["kubernetes"] = _to_json_dict(profile.kubernetes)

    # Top-level MinIO fields (snake_case, matching Java format)
    for minio_field in ("minio_endpoint", "minio_access_key",
                        "minio_secret_key", "minio_bucket"):
        value = getattr(profile, minio_field)
        if value is not None:
            data[minio_field] = value

    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def load_target(name: str) -> TargetProfile:
    """Read a target profile from ``~/.josh/targets/<name>.json``.

    Args:
        name: Profile name.

    Returns:
        Parsed :class:`TargetProfile`.

    Raises:
        FileNotFoundError: If profile does not exist.
        ValueError: If JSON is malformed or missing required fields.
    """
    path = _profile_path(name)
    raw = json.loads(path.read_text())

    target_type = raw.get("type")
    if target_type not in ("http", "kubernetes"):
        raise ValueError(
            f"Invalid or missing 'type' in {path}: got {target_type!r}, "
            "expected 'http' or 'kubernetes'."
        )

    http_config = None
    k8s_config = None

    if target_type == "http" and "http" in raw:
        http_config = HttpTargetConfig(**_from_json_dict(raw["http"]))
    elif target_type == "kubernetes" and "kubernetes" in raw:
        k8s_config = KubernetesTargetConfig(**_from_json_dict(raw["kubernetes"]))

    return TargetProfile(
        target_type=target_type,
        http=http_config,
        kubernetes=k8s_config,
        minio_endpoint=raw.get("minio_endpoint"),
        minio_access_key=raw.get("minio_access_key"),
        minio_secret_key=raw.get("minio_secret_key"),
        minio_bucket=raw.get("minio_bucket"),
    )


def list_targets() -> list[str]:
    """List available target profile names.

    Returns:
        Sorted list of profile names (without ``.json`` extension).
        Empty list if ``~/.josh/targets/`` does not exist.
    """
    if not TARGETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in TARGETS_DIR.glob("*.json"))


def delete_target(name: str) -> None:
    """Remove a target profile.

    Args:
        name: Profile name.

    Raises:
        FileNotFoundError: If profile does not exist.
    """
    _profile_path(name).unlink()


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def resolve_minio_creds(
    target: TargetProfile | None = None,
) -> ResolvedMinioCreds:
    """Resolve MinIO credentials from profile + environment variables.

    Hierarchy (per field): profile JSON field > environment variable.

    Environment variables checked:
        ``MINIO_ENDPOINT``, ``MINIO_ACCESS_KEY``, ``MINIO_SECRET_KEY``,
        ``MINIO_BUCKET``.

    Args:
        target: Optional target profile with MinIO fields.

    Returns:
        :class:`ResolvedMinioCreds` with merged credentials.  Fields may
        still be ``None`` if neither source provides a value.
    """
    return ResolvedMinioCreds(
        endpoint=(
            (target.minio_endpoint if target else None)
            or os.environ.get("MINIO_ENDPOINT")
        ),
        access_key=(
            (target.minio_access_key if target else None)
            or os.environ.get("MINIO_ACCESS_KEY")
        ),
        secret_key=(
            (target.minio_secret_key if target else None)
            or os.environ.get("MINIO_SECRET_KEY")
        ),
        bucket=(
            (target.minio_bucket if target else None)
            or os.environ.get("MINIO_BUCKET")
        ),
    )
