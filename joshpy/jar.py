"""Josh JAR management utilities.

This module provides utilities for managing Josh JAR files:
- Downloading prod/dev jars from joshsim.org
- Selecting between prod, dev, and local jars
- Getting jar version information

Example usage:
    from joshpy.jar import JarManager, JarMode

    # Use production jar (downloads if needed)
    manager = JarManager()
    jar_path = manager.get_jar(JarMode.PROD)

    # Use development jar
    jar_path = manager.get_jar(JarMode.DEV)

    # Use local/custom jar
    jar_path = manager.get_jar(JarMode.LOCAL, local_path=Path("my-custom.jar"))
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import requests


class JarMode(Enum):
    """JAR selection mode."""

    PROD = "prod"
    DEV = "dev"
    LOCAL = "local"


# Download URLs for official jars
JAR_URLS = {
    JarMode.PROD: "https://joshsim.org/dist/main/joshsim-fat.jar",
    JarMode.DEV: "https://joshsim.org/dist/dev/joshsim-fat.jar",
}

# Default jar directory (relative to working directory)
DEFAULT_JAR_DIR = Path("jar")

# Default jar filenames
JAR_FILENAMES = {
    JarMode.PROD: "joshsim-fat-prod.jar",
    JarMode.DEV: "joshsim-fat-dev.jar",
    JarMode.LOCAL: "joshsim-fat.jar",
}


def get_jar_hash(jar_path: Path) -> str | None:
    """Get SHA256 hash of a jar file.

    Args:
        jar_path: Path to the jar file.

    Returns:
        Hex digest of SHA256 hash, or None if file doesn't exist.
    """
    if not jar_path.exists():
        return None

    sha256 = hashlib.sha256()
    with open(jar_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_jar_version(jar_path: Path, java_path: str = "java") -> str | None:
    """Get version string from a jar file.

    Args:
        jar_path: Path to the jar file.
        java_path: Path to java executable.

    Returns:
        Version string, or None if unable to determine.
    """
    if not jar_path.exists():
        return None

    try:
        result = subprocess.run(
            [java_path, "-jar", str(jar_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


@dataclass
class DownloadResult:
    """Result of a jar download operation."""

    jar_path: Path
    success: bool
    was_updated: bool
    old_hash: str | None
    new_hash: str | None
    old_version: str | None
    new_version: str | None
    error: str | None = None


class JarManager:
    """Manages Josh JAR files.

    Handles downloading, caching, and selecting between prod/dev/local jars.

    Attributes:
        jar_dir: Directory where jars are stored.
        java_path: Path to java executable.
    """

    def __init__(
        self,
        jar_dir: Path | None = None,
        java_path: str = "java",
    ):
        """Initialize JarManager.

        Args:
            jar_dir: Directory for storing jars. Defaults to ./jar/
            java_path: Path to java executable.
        """
        self.jar_dir = jar_dir or DEFAULT_JAR_DIR
        self.java_path = java_path

    def get_jar_path(self, mode: JarMode) -> Path:
        """Get the path where a jar would be stored.

        Args:
            mode: The jar mode (PROD, DEV, or LOCAL).

        Returns:
            Path to the jar file.
        """
        return self.jar_dir / JAR_FILENAMES[mode]

    def jar_exists(self, mode: JarMode) -> bool:
        """Check if a jar exists.

        Args:
            mode: The jar mode.

        Returns:
            True if the jar file exists.
        """
        return self.get_jar_path(mode).exists()

    def download_jar(
        self,
        mode: JarMode,
        force: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> DownloadResult:
        """Download a jar from joshsim.org.

        Args:
            mode: The jar mode (PROD or DEV). LOCAL mode raises ValueError.
            force: If True, download even if jar already exists.
            on_progress: Optional callback(bytes_downloaded, total_bytes).

        Returns:
            DownloadResult with download status.

        Raises:
            ValueError: If mode is LOCAL (can't download local jars).
        """
        if mode == JarMode.LOCAL:
            raise ValueError("Cannot download LOCAL jar - use a custom path instead")

        url = JAR_URLS[mode]
        jar_path = self.get_jar_path(mode)

        # Get current state
        old_hash = get_jar_hash(jar_path)
        old_version = get_jar_version(jar_path, self.java_path) if old_hash else None

        # Skip if exists and not forcing
        if old_hash and not force:
            return DownloadResult(
                jar_path=jar_path,
                success=True,
                was_updated=False,
                old_hash=old_hash,
                new_hash=old_hash,
                old_version=old_version,
                new_version=old_version,
            )

        # Ensure directory exists
        self.jar_dir.mkdir(parents=True, exist_ok=True)

        # Download
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(jar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress and total_size:
                        on_progress(downloaded, total_size)

        except requests.RequestException as e:
            return DownloadResult(
                jar_path=jar_path,
                success=False,
                was_updated=False,
                old_hash=old_hash,
                new_hash=None,
                old_version=old_version,
                new_version=None,
                error=str(e),
            )

        # Get new state
        new_hash = get_jar_hash(jar_path)
        new_version = get_jar_version(jar_path, self.java_path)

        return DownloadResult(
            jar_path=jar_path,
            success=True,
            was_updated=(old_hash != new_hash),
            old_hash=old_hash,
            new_hash=new_hash,
            old_version=old_version,
            new_version=new_version,
        )

    def get_jar(
        self,
        mode: JarMode = JarMode.PROD,
        local_path: Path | None = None,
        auto_download: bool = True,
    ) -> Path:
        """Get path to a jar, downloading if necessary.

        Args:
            mode: The jar mode.
            local_path: Custom path for LOCAL mode. Ignored for other modes.
            auto_download: If True, download prod/dev jars if not present.

        Returns:
            Path to the jar file.

        Raises:
            FileNotFoundError: If jar doesn't exist and can't be downloaded.
        """
        if mode == JarMode.LOCAL:
            if local_path is None:
                # Default local path
                local_path = self.get_jar_path(JarMode.LOCAL)

            if not local_path.exists():
                raise FileNotFoundError(
                    f"Local jar not found: {local_path}\n"
                    "Copy your custom jar to this location or specify a different path."
                )
            return local_path

        jar_path = self.get_jar_path(mode)

        if not jar_path.exists():
            if auto_download:
                result = self.download_jar(mode)
                if not result.success:
                    raise FileNotFoundError(
                        f"Failed to download {mode.value} jar: {result.error}\n"
                        f"URL: {JAR_URLS[mode]}"
                    )
            else:
                raise FileNotFoundError(
                    f"{mode.value.capitalize()} jar not found: {jar_path}\n"
                    f"Run 'pixi run get-jars' or download from: {JAR_URLS[mode]}"
                )

        return jar_path

    def ensure_jars(
        self,
        modes: list[JarMode] | None = None,
        force: bool = False,
    ) -> dict[JarMode, DownloadResult]:
        """Ensure specified jars are downloaded.

        Args:
            modes: List of modes to download. Defaults to [PROD, DEV].
            force: If True, re-download even if jars exist.

        Returns:
            Dict mapping mode to download result.
        """
        if modes is None:
            modes = [JarMode.PROD, JarMode.DEV]

        results = {}
        for mode in modes:
            if mode != JarMode.LOCAL:
                results[mode] = self.download_jar(mode, force=force)

        return results

    def get_info(self, mode: JarMode) -> dict[str, str | None]:
        """Get information about a jar.

        Args:
            mode: The jar mode.

        Returns:
            Dict with 'path', 'exists', 'version', 'hash' keys.
        """
        jar_path = self.get_jar_path(mode)
        exists = jar_path.exists()

        return {
            "path": str(jar_path),
            "exists": exists,
            "version": get_jar_version(jar_path, self.java_path) if exists else None,
            "hash": get_jar_hash(jar_path)[:16] + "..." if exists else None,
        }


# Convenience functions for common operations


def download_jars(
    jar_dir: Path | None = None,
    force: bool = False,
) -> dict[JarMode, DownloadResult]:
    """Download prod and dev jars.

    Convenience function equivalent to JarManager().ensure_jars().

    Args:
        jar_dir: Directory for storing jars.
        force: If True, re-download even if jars exist.

    Returns:
        Dict mapping mode to download result.
    """
    manager = JarManager(jar_dir=jar_dir)
    return manager.ensure_jars(force=force)


def get_jar(
    mode: JarMode = JarMode.PROD,
    local_path: Path | None = None,
    jar_dir: Path | None = None,
    auto_download: bool = True,
) -> Path:
    """Get path to a jar, downloading if necessary.

    Convenience function equivalent to JarManager().get_jar().

    Args:
        mode: The jar mode (PROD, DEV, or LOCAL).
        local_path: Custom path for LOCAL mode.
        jar_dir: Directory for storing jars.
        auto_download: If True, download if not present.

    Returns:
        Path to the jar file.
    """
    manager = JarManager(jar_dir=jar_dir)
    return manager.get_jar(mode=mode, local_path=local_path, auto_download=auto_download)
