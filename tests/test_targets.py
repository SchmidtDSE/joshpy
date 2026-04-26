"""Tests for joshpy.targets — target profile system."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from joshpy.targets import (
    HttpTargetConfig,
    KubernetesTargetConfig,
    ResolvedMinioCreds,
    TargetProfile,
    _from_json_dict,
    _to_json_dict,
    delete_target,
    list_targets,
    load_target,
    resolve_minio_creds,
    save_target,
)


# -----------------------------------------------------------------------
# Dataclass construction and validation
# -----------------------------------------------------------------------


class TestHttpTargetConfig(unittest.TestCase):
    def test_required_fields(self):
        cfg = HttpTargetConfig(endpoint="https://example.com")
        self.assertEqual(cfg.endpoint, "https://example.com")

    def test_defaults(self):
        cfg = HttpTargetConfig(endpoint="https://example.com")
        self.assertIsNone(cfg.api_key)

    def test_frozen(self):
        cfg = HttpTargetConfig(endpoint="https://example.com")
        with self.assertRaises(AttributeError):
            cfg.endpoint = "other"  # type: ignore[misc]


class TestKubernetesTargetConfig(unittest.TestCase):
    def test_required_fields(self):
        cfg = KubernetesTargetConfig(namespace="josh", image="img:latest")
        self.assertEqual(cfg.namespace, "josh")
        self.assertEqual(cfg.image, "img:latest")

    def test_defaults(self):
        cfg = KubernetesTargetConfig(namespace="josh", image="img:latest")
        self.assertIsNone(cfg.context)
        self.assertIsNone(cfg.pod_minio_endpoint)
        self.assertEqual(cfg.resources, {})
        self.assertIsNone(cfg.parallelism)
        self.assertIsNone(cfg.timeout_seconds)
        self.assertIsNone(cfg.ttl_seconds_after_finished)
        self.assertFalse(cfg.spot)

    def test_resources_dict_isolation(self):
        a = KubernetesTargetConfig(namespace="a", image="a")
        b = KubernetesTargetConfig(namespace="b", image="b")
        self.assertIsNot(a.resources, b.resources)

    def test_frozen(self):
        cfg = KubernetesTargetConfig(namespace="josh", image="img:latest")
        with self.assertRaises(AttributeError):
            cfg.namespace = "other"  # type: ignore[misc]


class TestTargetProfile(unittest.TestCase):
    def test_http_profile(self):
        p = TargetProfile(
            target_type="http",
            http=HttpTargetConfig(endpoint="https://example.com"),
        )
        self.assertEqual(p.target_type, "http")
        self.assertIsNotNone(p.http)
        self.assertIsNone(p.kubernetes)

    def test_k8s_profile(self):
        p = TargetProfile(
            target_type="kubernetes",
            kubernetes=KubernetesTargetConfig(namespace="ns", image="img"),
        )
        self.assertEqual(p.target_type, "kubernetes")
        self.assertIsNone(p.http)
        self.assertIsNotNone(p.kubernetes)

    def test_http_requires_config(self):
        with self.assertRaises(ValueError, msg="http config required"):
            TargetProfile(target_type="http")

    def test_k8s_requires_config(self):
        with self.assertRaises(ValueError, msg="kubernetes config required"):
            TargetProfile(target_type="kubernetes")

    def test_minio_fields(self):
        p = TargetProfile(
            target_type="http",
            http=HttpTargetConfig(endpoint="https://example.com"),
            minio_endpoint="https://storage.googleapis.com",
            minio_bucket="my-bucket",
        )
        self.assertEqual(p.minio_endpoint, "https://storage.googleapis.com")
        self.assertEqual(p.minio_bucket, "my-bucket")
        self.assertIsNone(p.minio_access_key)
        self.assertIsNone(p.minio_secret_key)


# -----------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------


class TestSerialization(unittest.TestCase):
    def test_to_json_renames_keys(self):
        cfg = HttpTargetConfig(endpoint="https://ex.com", api_key="secret")
        d = _to_json_dict(cfg)
        self.assertIn("apiKey", d)
        self.assertNotIn("api_key", d)
        self.assertEqual(d["apiKey"], "secret")

    def test_to_json_omits_none(self):
        cfg = HttpTargetConfig(endpoint="https://ex.com")
        d = _to_json_dict(cfg)
        self.assertNotIn("apiKey", d)
        self.assertNotIn("api_key", d)

    def test_from_json_renames_keys(self):
        d = _from_json_dict({"apiKey": "secret", "endpoint": "https://ex.com"})
        self.assertIn("api_key", d)
        self.assertNotIn("apiKey", d)
        self.assertEqual(d["api_key"], "secret")

    def test_roundtrip_k8s(self):
        original = KubernetesTargetConfig(
            namespace="ns",
            image="img:latest",
            context="gke_proj_region_cluster",
            pod_minio_endpoint="https://storage.googleapis.com",
            resources={"requests": {"cpu": "1", "memory": "2Gi"}},
            parallelism=5,
            timeout_seconds=600,
            ttl_seconds_after_finished=3600,
            spot=True,
        )
        json_dict = _to_json_dict(original)
        python_dict = _from_json_dict(json_dict)
        restored = KubernetesTargetConfig(**python_dict)
        self.assertEqual(restored, original)


# -----------------------------------------------------------------------
# Save / load round-trips (filesystem)
# -----------------------------------------------------------------------


class TestSaveLoadTarget(unittest.TestCase):
    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "nested" / "targets"
            with patch("joshpy.targets.TARGETS_DIR", target_dir):
                profile = TargetProfile(
                    target_type="http",
                    http=HttpTargetConfig(endpoint="https://ex.com"),
                )
                path = save_target("test", profile)
                self.assertTrue(path.exists())
                self.assertTrue(target_dir.is_dir())

    def test_roundtrip_http(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("joshpy.targets.TARGETS_DIR", Path(tmpdir)):
                original = TargetProfile(
                    target_type="http",
                    http=HttpTargetConfig(
                        endpoint="https://josh.example.com",
                        api_key="sk-123",
                    ),
                    minio_endpoint="https://storage.googleapis.com",
                    minio_bucket="josh-bucket",
                )
                save_target("cloud-dev", original)
                loaded = load_target("cloud-dev")
                self.assertEqual(loaded.target_type, "http")
                self.assertEqual(loaded.http, original.http)
                self.assertEqual(loaded.minio_endpoint, original.minio_endpoint)
                self.assertEqual(loaded.minio_bucket, original.minio_bucket)

    def test_roundtrip_k8s(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("joshpy.targets.TARGETS_DIR", Path(tmpdir)):
                original = TargetProfile(
                    target_type="kubernetes",
                    kubernetes=KubernetesTargetConfig(
                        namespace="joshsim",
                        image="ghcr.io/schmidtdse/joshsim:latest",
                        context="gke_proj_us-west1_cluster",
                        pod_minio_endpoint="https://storage.googleapis.com",
                        resources={
                            "requests": {"cpu": "1", "memory": "2Gi"},
                            "limits": {"memory": "4Gi"},
                        },
                        parallelism=5,
                        timeout_seconds=600,
                        ttl_seconds_after_finished=3600,
                        spot=True,
                    ),
                    minio_endpoint="https://storage.googleapis.com",
                    minio_bucket="josh-bucket",
                )
                save_target("gke-test", original)
                loaded = load_target("gke-test")
                self.assertEqual(loaded.target_type, "kubernetes")
                self.assertEqual(loaded.kubernetes, original.kubernetes)
                self.assertEqual(loaded.minio_endpoint, original.minio_endpoint)

    def test_roundtrip_minio_creds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("joshpy.targets.TARGETS_DIR", Path(tmpdir)):
                original = TargetProfile(
                    target_type="http",
                    http=HttpTargetConfig(endpoint="https://ex.com"),
                    minio_endpoint="https://storage.googleapis.com",
                    minio_access_key="GOOG123",
                    minio_secret_key="secret456",
                    minio_bucket="my-bucket",
                )
                save_target("creds-test", original)
                loaded = load_target("creds-test")
                self.assertEqual(loaded.minio_access_key, "GOOG123")
                self.assertEqual(loaded.minio_secret_key, "secret456")

    def test_load_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("joshpy.targets.TARGETS_DIR", Path(tmpdir)):
                with self.assertRaises(FileNotFoundError):
                    load_target("does-not-exist")

    def test_json_uses_correct_keys(self):
        """Verify the raw JSON matches the format joshsim expects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("joshpy.targets.TARGETS_DIR", Path(tmpdir)):
                profile = TargetProfile(
                    target_type="kubernetes",
                    kubernetes=KubernetesTargetConfig(
                        namespace="ns",
                        image="img",
                        timeout_seconds=600,
                        ttl_seconds_after_finished=3600,
                    ),
                    minio_endpoint="https://storage.googleapis.com",
                    minio_bucket="bucket",
                )
                path = save_target("keys-test", profile)
                raw = json.loads(path.read_text())

                # Top-level keys
                self.assertEqual(raw["type"], "kubernetes")
                self.assertEqual(raw["minio_endpoint"], "https://storage.googleapis.com")
                self.assertEqual(raw["minio_bucket"], "bucket")

                # Nested K8s keys — camelCase where Java expects it
                k8s = raw["kubernetes"]
                self.assertIn("timeoutSeconds", k8s)
                self.assertIn("ttlSecondsAfterFinished", k8s)
                # snake_case fields stay snake_case
                self.assertNotIn("timeout_seconds", k8s)
                self.assertNotIn("ttl_seconds_after_finished", k8s)

    def test_invalid_name_rejected(self):
        with self.assertRaises(ValueError, msg="Invalid target name"):
            save_target("../etc/passwd", TargetProfile(
                target_type="http",
                http=HttpTargetConfig(endpoint="https://ex.com"),
            ))

        with self.assertRaises(ValueError):
            save_target("has.dot", TargetProfile(
                target_type="http",
                http=HttpTargetConfig(endpoint="https://ex.com"),
            ))

        with self.assertRaises(ValueError):
            save_target("has spaces", TargetProfile(
                target_type="http",
                http=HttpTargetConfig(endpoint="https://ex.com"),
            ))


# -----------------------------------------------------------------------
# List / delete
# -----------------------------------------------------------------------


class TestListDeleteTargets(unittest.TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("joshpy.targets.TARGETS_DIR", Path(tmpdir)):
                self.assertEqual(list_targets(), [])

    def test_multiple_targets_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("joshpy.targets.TARGETS_DIR", Path(tmpdir)):
                for name in ("zebra", "alpha", "middle"):
                    save_target(name, TargetProfile(
                        target_type="http",
                        http=HttpTargetConfig(endpoint="https://ex.com"),
                    ))
                self.assertEqual(list_targets(), ["alpha", "middle", "zebra"])

    def test_no_dir_exists(self):
        with patch("joshpy.targets.TARGETS_DIR", Path("/tmp/nonexistent-josh-targets-abc")):
            self.assertEqual(list_targets(), [])

    def test_delete_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("joshpy.targets.TARGETS_DIR", Path(tmpdir)):
                save_target("to-delete", TargetProfile(
                    target_type="http",
                    http=HttpTargetConfig(endpoint="https://ex.com"),
                ))
                self.assertIn("to-delete", list_targets())
                delete_target("to-delete")
                self.assertNotIn("to-delete", list_targets())

    def test_delete_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("joshpy.targets.TARGETS_DIR", Path(tmpdir)):
                with self.assertRaises(FileNotFoundError):
                    delete_target("nope")


# -----------------------------------------------------------------------
# Credential resolution
# -----------------------------------------------------------------------


class TestResolveMinioCredentials(unittest.TestCase):
    def test_no_target_uses_env(self):
        env = {
            "MINIO_ENDPOINT": "https://env.example.com",
            "MINIO_ACCESS_KEY": "env-ak",
            "MINIO_SECRET_KEY": "env-sk",
            "MINIO_BUCKET": "env-bucket",
        }
        with patch.dict(os.environ, env, clear=False):
            creds = resolve_minio_creds()
            self.assertEqual(creds.endpoint, "https://env.example.com")
            self.assertEqual(creds.access_key, "env-ak")
            self.assertEqual(creds.secret_key, "env-sk")
            self.assertEqual(creds.bucket, "env-bucket")

    def test_profile_overrides_env(self):
        env = {
            "MINIO_ENDPOINT": "https://env.example.com",
            "MINIO_BUCKET": "env-bucket",
        }
        profile = TargetProfile(
            target_type="http",
            http=HttpTargetConfig(endpoint="https://ex.com"),
            minio_endpoint="https://profile.example.com",
            minio_bucket="profile-bucket",
        )
        with patch.dict(os.environ, env, clear=False):
            creds = resolve_minio_creds(profile)
            self.assertEqual(creds.endpoint, "https://profile.example.com")
            self.assertEqual(creds.bucket, "profile-bucket")

    def test_partial_merge(self):
        env = {
            "MINIO_ACCESS_KEY": "env-ak",
            "MINIO_SECRET_KEY": "env-sk",
        }
        profile = TargetProfile(
            target_type="http",
            http=HttpTargetConfig(endpoint="https://ex.com"),
            minio_endpoint="https://profile.example.com",
            minio_bucket="profile-bucket",
        )
        with patch.dict(os.environ, env, clear=False):
            creds = resolve_minio_creds(profile)
            self.assertEqual(creds.endpoint, "https://profile.example.com")
            self.assertEqual(creds.access_key, "env-ak")
            self.assertEqual(creds.secret_key, "env-sk")
            self.assertEqual(creds.bucket, "profile-bucket")

    def test_no_source_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            creds = resolve_minio_creds()
            self.assertIsNone(creds.endpoint)
            self.assertIsNone(creds.access_key)
            self.assertIsNone(creds.secret_key)
            self.assertIsNone(creds.bucket)

    def test_none_target_uses_env(self):
        env = {"MINIO_ENDPOINT": "https://env.example.com"}
        with patch.dict(os.environ, env, clear=False):
            creds = resolve_minio_creds(target=None)
            self.assertEqual(creds.endpoint, "https://env.example.com")


if __name__ == "__main__":
    unittest.main()
