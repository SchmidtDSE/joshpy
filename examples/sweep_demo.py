#!/usr/bin/env python3
"""Demo script showing how to run a parameter sweep with joshpy.

This script demonstrates the Jinja templating workflow:
1. Define a template for .jshc configuration files
2. Define parameter values to sweep over
3. Expand the template for each parameter combination
4. Execute jobs against the Josh runtime

Prerequisites:
    pip install joshpy[jobs]
    # Ensure joshsim-fat.jar is built: gradle fatJar

Usage:
    python sweep_demo.py [--jar PATH_TO_JAR]
"""

import argparse
from pathlib import Path

# Import joshpy jobs module
from joshpy.jobs import (
    JobConfig,
    SweepConfig,
    SweepParameter,
    JobExpander,
    JobRunner,
    run_sweep,
)


def main():
    parser = argparse.ArgumentParser(description="Run a parameter sweep demo")
    parser.add_argument(
        "--jar",
        type=Path,
        default=None,
        help="Path to joshsim-fat.jar (download from https://language.joshsim.org/download.html)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    # Paths relative to this script
    script_dir = Path(__file__).parent
    source_path = script_dir / "hello_cli_configurable.josh"
    template_path = script_dir / "templates/editor.jshc.j2"

    # Verify files exist
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not args.dry_run:
        if args.jar is None:
            raise ValueError(
                "JAR path required. Use --jar PATH or download from:\n"
                "https://language.joshsim.org/download.html"
            )
        if not args.jar.exists():
            raise FileNotFoundError(
                f"JAR file not found: {args.jar}\n"
                "Download from: https://language.joshsim.org/download.html"
            )

    print("=" * 60)
    print("joshpy Parameter Sweep Demo")
    print("=" * 60)
    print(f"Source:   {source_path}")
    print(f"Template: {template_path}")
    print(f"JAR:      {args.jar}")
    print()

    # ==================================================================
    # Method 1: Using JobConfig + JobExpander + JobRunner (full control)
    # ==================================================================
    print("Method 1: Full JobConfig workflow")
    print("-" * 40)

    # Define the job configuration
    config = JobConfig(
        template_path=template_path,
        source_path=source_path,
        simulation="Main",
        replicates=1,
        sweep=SweepConfig(
            parameters=[
                # Sweep over different max growth values
                SweepParameter(name="maxGrowth", values=[1, 5, 10]),
            ]
        ),
    )

    # Show YAML representation
    print("Job Config (YAML):")
    print(config.to_yaml())

    # Expand to concrete jobs
    expander = JobExpander()
    job_set = expander.expand(config)

    print(f"Generated {len(job_set)} jobs:")
    for i, job in enumerate(job_set.jobs):
        print(f"  Job {i}: maxGrowth={job.parameters.get('maxGrowth')}")
        print(f"    Config hash: {job.config_hash}")
        print(f"    Config path: {job.config_path}")
        print(f"    Content:\n      {job.config_content.strip()}")
        print()

    if args.dry_run:
        print("DRY RUN - Commands that would be executed:")
        runner = JobRunner(josh_jar=args.jar)
        for job in job_set.jobs:
            cmd = runner.build_command(job)
            print(f"  {' '.join(cmd)}")
        job_set.cleanup()
    else:
        # Execute jobs
        print("Executing jobs...")
        runner = JobRunner(josh_jar=args.jar)

        def on_complete(result):
            status = "OK" if result.success else "FAIL"
            params = result.job.parameters
            print(f"  [{status}] maxGrowth={params.get('maxGrowth')}")
            if not result.success:
                print(f"    Error: {result.stderr[:200]}")

        results = runner.run_all(job_set, on_complete=on_complete)

        # Summary
        print()
        print("Summary:")
        succeeded = sum(1 for r in results if r.success)
        print(f"  {succeeded}/{len(results)} jobs succeeded")

    print()
    print("=" * 60)

    # ==================================================================
    # Method 2: Using run_sweep() convenience function
    # ==================================================================
    print("Method 2: Using run_sweep() convenience function")
    print("-" * 40)

    print("Code example:")
    print('''
    from joshpy.jobs import run_sweep

    results = run_sweep(
        template=Path("templates/editor.jshc.j2"),
        source=Path("hello_cli_configurable.josh"),
        parameters={"maxGrowth": [1, 5, 10]},
        josh_jar=Path("joshsim-fat.jar"),
    )
    ''')

    if not args.dry_run:
        results = run_sweep(
            template=template_path,
            source=source_path,
            parameters={"maxGrowth": [1, 5, 10]},
            josh_jar=args.jar,
        )
        print(f"Results: {sum(1 for r in results if r.success)}/{len(results)} succeeded")

    print()
    print("=" * 60)

    # ==================================================================
    # Method 3: Using inline template string
    # ==================================================================
    print("Method 3: Using inline template string")
    print("-" * 40)

    config_inline = JobConfig(
        template_string="maxGrowth = {{ maxGrowth }} meters",
        source_path=source_path,
        sweep=SweepConfig(
            parameters=[
                SweepParameter(name="maxGrowth", values=[2, 4, 8]),
            ]
        ),
    )

    job_set_inline = expander.expand(config_inline)
    print(f"Generated {len(job_set_inline)} jobs with inline template")
    for job in job_set_inline.jobs:
        print(f"  maxGrowth={job.parameters['maxGrowth']}: '{job.config_content}'")

    job_set_inline.cleanup()
    print()

    # ==================================================================
    # Method 4: Multi-parameter sweep (cartesian product)
    # ==================================================================
    print("Method 4: Multi-parameter sweep (cartesian product)")
    print("-" * 40)

    # This demonstrates a 3x3 = 9 combination sweep
    # (not actually valid for this demo since we only have maxGrowth,
    # but shows the syntax)
    multi_config = JobConfig(
        template_string="param1 = {{ p1 }}\nparam2 = {{ p2 }}",
        sweep=SweepConfig(
            parameters=[
                SweepParameter(name="p1", values=[10, 20, 30]),
                SweepParameter(name="p2", values=["low", "high", "extreme"]),
            ]
        ),
    )

    multi_set = expander.expand(multi_config)
    print(f"Generated {len(multi_set)} jobs (3 x 3 = 9)")
    for job in multi_set.jobs:
        print(f"  p1={job.parameters['p1']}, p2={job.parameters['p2']}")

    multi_set.cleanup()
    print()

    # ==================================================================
    # Method 5: Using range specifications (YAML style)
    # ==================================================================
    print("Method 5: Range specifications")
    print("-" * 40)

    # Range with step (like numpy.arange)
    range_step = SweepParameter(
        name="survivalProb",
        values={"start": 80, "stop": 100, "step": 5}
    )
    print(f"Range with step: {range_step.values}")

    # Range with count (like numpy.linspace)
    range_num = SweepParameter(
        name="seedCount",
        values={"start": 1000, "stop": 5000, "num": 5}
    )
    print(f"Range with num:  {range_num.values}")

    print()
    print("=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
