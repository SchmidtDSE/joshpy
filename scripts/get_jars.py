#!/usr/bin/env python3
"""Download/update Josh JAR files."""

from joshpy.jar import download_jars


def main():
    print("Checking for JAR updates...")
    # force=True to always download and compare hashes
    results = download_jars(force=True)

    for mode, result in results.items():
        version = result.new_version or "unknown"
        if result.was_updated:
            old_ver = result.old_version or "none"
            old_hash = (result.old_hash[:12] + "...") if result.old_hash else "none"
            new_hash = (result.new_hash[:12] + "...") if result.new_hash else "unknown"
            print(f"  {mode.value}: Updated ({old_ver} -> {version})")
            print(f"           hash: {old_hash} -> {new_hash}")
        else:
            print(f"  {mode.value}: Up to date ({version})")


if __name__ == "__main__":
    main()
