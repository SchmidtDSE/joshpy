"""Per-job staging directory assembly for batch remote execution.

The sweep loop assembles one directory per ``ExpandedJob`` so each job stages
only its own ``.josh``/``.jshc``/``.jshd`` files into its MinIO prefix. This
avoids the directory-contamination bug where sibling fixture files were swept
in when the pre-#423 JAR staged a whole parent directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from joshpy.jobs import ExpandedJob


def assemble_batch_workdir(job: ExpandedJob, workdir: Path) -> Path:
    """Create a per-ExpandedJob staging directory.

    Layout::

        workdir/<run_hash>/
          sim.josh            # symlink to job.source_path
          config.jshc         # unique rendered config for this job (written)
          <name>.jshd         # symlink per entry in job.file_mappings

    Uses symlinks (not copies) to avoid disk duplication of large .jshd files;
    ``cli.stage_to_minio`` uploads via content reads and follows symlinks.
    Idempotent: re-running against the same workdir replaces existing entries.

    Args:
        job: The expanded job. ``job.source_path`` must be set.
        workdir: Parent directory to create the per-job subdir inside.

    Returns:
        Path to the per-job staging directory (``workdir/<run_hash>/``).

    Raises:
        ValueError: If ``job.source_path`` is None.
    """
    if job.source_path is None:
        raise ValueError("ExpandedJob.source_path is required for batch remote")

    target = workdir / job.run_hash
    target.mkdir(parents=True, exist_ok=True)

    sim_link = target / "sim.josh"
    if sim_link.exists() or sim_link.is_symlink():
        sim_link.unlink()
    os.symlink(job.source_path.resolve(), sim_link)

    (target / "config.jshc").write_text(job.config_content)

    for name, path in job.file_mappings.items():
        dest_name = name if name.endswith(".jshd") else f"{name}.jshd"
        dest = target / dest_name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(path.resolve(), dest)

    return target
