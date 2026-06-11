"""Josh JAR management utilities.

Downloads, caches, and version-checks the Josh simulation JAR (prod/dev/local).

CLI usage::

    python -m joshpy.jar            # download/refresh prod + dev jars
    python -m joshpy.jar --force    # re-download even if up to date
"""

from joshpy.jar._core import (  # noqa: F401
    DEFAULT_JAR_DIR,
    JAR_FILENAMES,
    JAR_URLS,
    DownloadResult,
    JarManager,
    JarMode,
    download_jars,
    get_jar,
    get_jar_hash,
    get_jar_version,
)
