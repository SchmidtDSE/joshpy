"""Diagnostic plotting for Josh simulation outputs.

Quick visualization primitives for simulation sanity checks:
- Are variables changing over time?
- Are values exploding or crashing?
- How do different parameter settings compare?

Example:
    from joshpy.registry import RunRegistry
    from joshpy.diagnostics import SimulationDiagnostics

    registry = RunRegistry("experiment.duckdb")
    diag = SimulationDiagnostics(registry)

    # What's in the data?
    print(registry.get_data_summary())

    # Plot time series (spatially aggregated)
    diag.plot_timeseries("averageAge", config_hash="abc123")

    # Plot per-patch time series (no aggregation)
    diag.plot_timeseries("averageAge", config_hash="abc123", aggregate="none")

    # Compare across parameter values
    diag.plot_comparison("averageAge", group_by="maxGrowth")

    # Filter by arbitrary parameters
    diag.plot_timeseries("averageAge", maxGrowth=10, survivalProb=0.9)
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt

from joshpy.registry import RunRegistry


def _sort_key_numeric_then_string(value: str) -> tuple[int, float | str]:
    """Sort key that orders numeric values numerically, then strings alphabetically.
    
    Returns a tuple of (type_order, value) where:
    - type_order=0 for numeric values (sorted by numeric value)
    - type_order=1 for string values (sorted alphabetically)
    
    This ensures: 1, 2, 10, 20, 100 instead of "1", "10", "100", "2", "20"
    """
    try:
        return (0, float(value))
    except (ValueError, TypeError):
        return (1, str(value))


class SimulationDiagnostics:
    """Diagnostic plotting backed by a RunRegistry.

    Provides quick visualization methods for simulation sanity checks.
    All plot methods display inline by default and return the matplotlib Figure
    for further customization or saving.

    Attributes:
        registry: The RunRegistry containing simulation data.

    Example:
        registry = RunRegistry("experiment.duckdb")
        diag = SimulationDiagnostics(registry)

        # Quick time series check
        diag.plot_timeseries("averageAge", config_hash="abc123")

        # Save to file
        fig = diag.plot_timeseries("averageAge", config_hash="abc123", show=False)
        fig.savefig("diagnostic.png")
    """

    # Valid aggregation modes
    VALID_AGGREGATIONS = {"none", "mean", "sum", "min", "max"}

    def __init__(self, registry: RunRegistry) -> None:
        """Initialize with a registry connection.

        Args:
            registry: RunRegistry instance containing simulation data.

        Raises:
            TypeError: If registry is not a RunRegistry instance.
        """
        if not isinstance(registry, RunRegistry):
            raise TypeError(f"Expected RunRegistry, got {type(registry)}")
        self.registry = registry

    def _validate_variable(self, variable: str) -> None:
        """Validate that a variable exists in the registry.

        Args:
            variable: Variable name to check.

        Raises:
            ValueError: If variable not found, with list of available variables.
        """
        available = self.registry.list_variables()
        if variable not in available:
            available_str = ", ".join(available) if available else "(none)"
            raise ValueError(
                f"Variable '{variable}' not found. Available: {available_str}"
            )

    def _validate_parameter(self, param_name: str) -> None:
        """Validate that a parameter exists in the registry.

        Args:
            param_name: Parameter name to check.

        Raises:
            ValueError: If parameter not found, with list of available parameters.
        """
        available = self.registry.list_parameters()
        if param_name not in available:
            available_str = ", ".join(available) if available else "(none)"
            raise ValueError(
                f"Parameter '{param_name}' not found. Available: {available_str}"
            )

    def _validate_aggregation(self, aggregate: str) -> None:
        """Validate aggregation mode.

        Args:
            aggregate: Aggregation mode to check.

        Raises:
            ValueError: If aggregate not in valid set.
        """
        if aggregate not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation '{aggregate}'. "
                f"Valid options: {', '.join(sorted(self.VALID_AGGREGATIONS))}"
            )

    def _build_config_filter(
        self,
        config_hash: str | None,
        session_id: str | None,
        params: dict[str, Any],
    ) -> tuple[str, list[Any]]:
        """Build SQL WHERE clause for filtering by config/session/params.

        Args:
            config_hash: Optional config hash filter.
            session_id: Optional session ID filter.
            params: Dict of parameter name -> value filters.

        Returns:
            Tuple of (WHERE clause string, list of parameter values).
        """
        conditions = []
        values: list[Any] = []

        if config_hash:
            conditions.append("cd.config_hash = ?")
            values.append(config_hash)

        if session_id:
            conditions.append("jc.session_id = ?")
            values.append(session_id)

        for param_name, param_value in params.items():
            conditions.append(
                f"json_extract_string(jc.parameters, '$.{param_name}') = ?"
            )
            values.append(str(param_value))

        if conditions:
            return " AND " + " AND ".join(conditions), values
        return "", values

    def _get_matching_configs(
        self,
        config_hash: str | None,
        session_id: str | None,
        params: dict[str, Any],
    ) -> list[str]:
        """Get list of config hashes matching the filters.

        Args:
            config_hash: Optional config hash filter.
            session_id: Optional session ID filter.
            params: Dict of parameter name -> value filters.

        Returns:
            List of matching config hashes.
        """
        where_clause, values = self._build_config_filter(config_hash, session_id, params)

        query = """
            SELECT DISTINCT cd.config_hash
            FROM cell_data cd
            JOIN job_configs jc ON cd.config_hash = jc.config_hash
            WHERE 1=1
        """
        query += where_clause

        result = self.registry.conn.execute(query, values).fetchall()
        return [row[0] for row in result]

    def plot_timeseries(
        self,
        variable: str,
        config_hash: str | None = None,
        session_id: str | None = None,
        aggregate: str = "mean",
        show_replicates: bool = True,
        step_range: tuple[int, int] | None = None,
        title: str | None = None,
        show: bool = True,
        **params: Any,
    ) -> plt.Figure:
        """Plot variable over time.

        Args:
            variable: Variable name to plot (e.g., "averageAge").
            config_hash: Filter to specific config.
            session_id: Filter to specific session.
            aggregate: Spatial aggregation mode: "mean" (default), "sum",
                "min", "max", or "none" (per-patch).
            show_replicates: If True and aggregate != "none", show uncertainty
                band across replicates.
            step_range: Optional (min_step, max_step) filter.
            title: Plot title (auto-generated if None).
            show: If True, display plot inline.
            **params: Filter by parameter values (e.g., maxGrowth=10).

        Returns:
            matplotlib Figure.

        Raises:
            ValueError: If variable not found or no data matches filters.
        """
        self._validate_variable(variable)
        self._validate_aggregation(aggregate)

        # Warn about aggregate="none"
        if aggregate == "none":
            warnings.warn(
                "aggregate='none' plots one line per patch, which may be busy "
                "for large grids. Lines are drawn with low alpha.",
                UserWarning,
                stacklevel=2,
            )

        # Get matching configs
        matching_configs = self._get_matching_configs(config_hash, session_id, params)

        if not matching_configs:
            raise ValueError(
                "No data found matching filters. "
                "Check session_id/config_hash/parameters."
            )

        if len(matching_configs) > 1:
            warnings.warn(
                f"Matched {len(matching_configs)} configs. "
                "Consider filtering by config_hash for clarity.",
                UserWarning,
                stacklevel=2,
            )

        # Build step filter
        step_filter = ""
        step_values: list[Any] = []
        if step_range:
            step_filter = " AND cd.step BETWEEN ? AND ?"
            step_values = [step_range[0], step_range[1]]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color cycle for multiple configs
        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        for idx, cfg_hash in enumerate(matching_configs):
            color = colors[idx % len(colors)]
            label = cfg_hash[:8] if len(matching_configs) > 1 else None

            if aggregate == "none":
                # Per-patch time series
                self._plot_per_patch(
                    ax, variable, cfg_hash, step_filter, step_values, color, label
                )
            else:
                # Aggregated time series
                self._plot_aggregated(
                    ax,
                    variable,
                    cfg_hash,
                    aggregate,
                    show_replicates,
                    step_filter,
                    step_values,
                    color,
                    label,
                )

        # Labels and title
        ax.set_xlabel("Step")
        ax.set_ylabel(variable)
        if title:
            ax.set_title(title)
        else:
            agg_label = f" ({aggregate})" if aggregate != "none" else " (per-patch)"
            ax.set_title(f"{variable} over time{agg_label}")

        if len(matching_configs) > 1:
            ax.legend()

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    def _plot_per_patch(
        self,
        ax: Any,
        variable: str,
        config_hash: str,
        step_filter: str,
        step_values: list[Any],
        color: Any,
        label: str | None,
    ) -> None:
        """Plot per-patch time series with low alpha."""
        query = f"""
            SELECT step, replicate, position_x, position_y,
                   CAST(json_extract_string(variables, '$.{variable}') AS DOUBLE) as value
            FROM cell_data cd
            WHERE cd.config_hash = ? {step_filter}
            ORDER BY position_x, position_y, replicate, step
        """
        values = [config_hash] + step_values
        result = self.registry.conn.execute(query, values).fetchall()

        if not result:
            return

        # Group by (position_x, position_y, replicate)
        from collections import defaultdict

        series: dict[tuple[float, float, int], tuple[list[int], list[float]]] = (
            defaultdict(lambda: ([], []))
        )
        for row in result:
            step, replicate, pos_x, pos_y, value = row
            key = (pos_x or 0.0, pos_y or 0.0, replicate)
            series[key][0].append(step)
            series[key][1].append(value)

        # Plot each series with low alpha
        for i, (key, (steps, values_list)) in enumerate(series.items()):
            # Only label the first line
            line_label = label if i == 0 else None
            ax.plot(steps, values_list, color=color, alpha=0.2, label=line_label)

    def _plot_aggregated(
        self,
        ax: Any,
        variable: str,
        config_hash: str,
        aggregate: str,
        show_replicates: bool,
        step_filter: str,
        step_values: list[Any],
        color: Any,
        label: str | None,
    ) -> None:
        """Plot spatially aggregated time series with optional replicate bands."""
        agg_func = aggregate.upper()

        # First aggregate spatially per step/replicate
        query = f"""
            SELECT step, replicate,
                   {agg_func}(CAST(json_extract_string(variables, '$.{variable}') AS DOUBLE)) as value
            FROM cell_data cd
            WHERE cd.config_hash = ? {step_filter}
            GROUP BY step, replicate
            ORDER BY step, replicate
        """
        values = [config_hash] + step_values
        result = self.registry.conn.execute(query, values).fetchall()

        if not result:
            return

        # Group by step to compute mean/std across replicates
        from collections import defaultdict

        step_data: dict[int, list[float]] = defaultdict(list)
        for row in result:
            step, replicate, value = row
            if value is not None:
                step_data[step].append(value)

        steps = sorted(step_data.keys())
        means = []
        stds = []
        for step in steps:
            vals = step_data[step]
            mean_val = sum(vals) / len(vals)
            means.append(mean_val)
            if len(vals) > 1:
                variance = sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
                stds.append(variance**0.5)
            else:
                stds.append(0.0)

        # Plot mean line
        ax.plot(steps, means, color=color, label=label, linewidth=2)

        # Plot uncertainty band if requested and we have multiple replicates
        if show_replicates and any(s > 0 for s in stds):
            lower = [m - s for m, s in zip(means, stds)]
            upper = [m + s for m, s in zip(means, stds)]
            ax.fill_between(steps, lower, upper, color=color, alpha=0.2)

    def plot_comparison(
        self,
        variable: str,
        group_by: str,
        aggregate: str = "mean",
        step: int | None = None,
        session_id: str | None = None,
        title: str | None = None,
        show: bool = True,
        **params: Any,
    ) -> plt.Figure:
        """Compare variable across parameter values.

        Args:
            variable: Variable name to plot.
            group_by: Parameter name to group by (e.g., "maxGrowth").
            aggregate: Spatial aggregation mode: "mean", "sum", "min", "max".
            step: If provided, creates bar chart at that step; else time series.
            session_id: Filter to specific session.
            title: Plot title.
            show: If True, display plot inline.
            **params: Additional parameter filters.

        Returns:
            matplotlib Figure.

        Raises:
            ValueError: If variable or parameter not found, or no data.
        """
        self._validate_variable(variable)
        self._validate_parameter(group_by)
        self._validate_aggregation(aggregate)

        if aggregate == "none":
            raise ValueError(
                "aggregate='none' is not supported for plot_comparison. "
                "Use 'mean', 'sum', 'min', or 'max'."
            )

        agg_func = aggregate.upper()

        # Build parameter filters
        param_conditions = []
        param_values: list[Any] = []

        if session_id:
            param_conditions.append("jc.session_id = ?")
            param_values.append(session_id)

        for param_name, param_value in params.items():
            param_conditions.append(
                f"json_extract_string(jc.parameters, '$.{param_name}') = ?"
            )
            param_values.append(str(param_value))

        where_clause = ""
        if param_conditions:
            where_clause = " AND " + " AND ".join(param_conditions)

        if step is not None:
            # Bar chart at specific step
            query = f"""
                SELECT
                    json_extract_string(jc.parameters, '$.{group_by}') as param_value,
                    {agg_func}(CAST(json_extract_string(cd.variables, '$.{variable}') AS DOUBLE)) as value
                FROM cell_data cd
                JOIN job_configs jc ON cd.config_hash = jc.config_hash
                WHERE cd.step = ? {where_clause}
                GROUP BY param_value
                ORDER BY param_value
            """
            values = [step] + param_values
            result = self.registry.conn.execute(query, values).fetchall()

            if not result:
                raise ValueError(
                    f"No data found at step {step} matching filters."
                )

            fig, ax = plt.subplots(figsize=(10, 6))

            # Sort results by parameter value (numeric then string)
            sorted_result = sorted(result, key=lambda r: _sort_key_numeric_then_string(str(r[0])))
            param_vals = [str(row[0]) for row in sorted_result]
            data_vals = [row[1] for row in sorted_result]

            ax.bar(param_vals, data_vals)
            ax.set_xlabel(group_by)
            ax.set_ylabel(f"{variable} ({aggregate})")
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"{variable} by {group_by} at step {step}")

        else:
            # Time series comparison
            query = f"""
                SELECT
                    json_extract_string(jc.parameters, '$.{group_by}') as param_value,
                    cd.step,
                    {agg_func}(CAST(json_extract_string(cd.variables, '$.{variable}') AS DOUBLE)) as value
                FROM cell_data cd
                JOIN job_configs jc ON cd.config_hash = jc.config_hash
                WHERE 1=1 {where_clause}
                GROUP BY param_value, cd.step
                ORDER BY param_value, cd.step
            """
            result = self.registry.conn.execute(query, param_values).fetchall()

            if not result:
                raise ValueError("No data found matching filters.")

            # Group by parameter value
            from collections import defaultdict

            param_data: dict[str, tuple[list[int], list[float]]] = defaultdict(
                lambda: ([], [])
            )
            for row in result:
                param_val, step_val, value = row
                param_data[str(param_val)][0].append(step_val)
                param_data[str(param_val)][1].append(value)

            fig, ax = plt.subplots(figsize=(10, 6))

            colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
            # Sort parameter values numerically then alphabetically
            sorted_params = sorted(param_data.keys(), key=_sort_key_numeric_then_string)
            for idx, param_val in enumerate(sorted_params):
                steps, values_list = param_data[param_val]
                color = colors[idx % len(colors)]
                ax.plot(steps, values_list, color=color, label=f"{group_by}={param_val}")

            ax.set_xlabel("Step")
            ax.set_ylabel(f"{variable} ({aggregate})")
            ax.legend()
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"{variable} by {group_by}")

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_spatial(
        self,
        variable: str,
        step: int,
        config_hash: str,
        replicate: int = 0,
        title: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot spatial scatter at a single timestep.

        Creates a scatter plot colored by variable value. For real GIS work,
        users should export to geopandas.

        Args:
            variable: Variable name to plot.
            step: Timestep to visualize.
            config_hash: Config hash to filter by.
            replicate: Replicate number (default: 0).
            title: Plot title.
            show: If True, display plot inline.

        Returns:
            matplotlib Figure.

        Raises:
            ValueError: If variable not found or no data matches.
        """
        self._validate_variable(variable)

        query = f"""
            SELECT
                longitude,
                latitude,
                CAST(json_extract_string(variables, '$.{variable}') AS DOUBLE) as value
            FROM cell_data
            WHERE config_hash = ?
              AND step = ?
              AND replicate = ?
              AND longitude IS NOT NULL
        """
        result = self.registry.conn.execute(
            query, [config_hash, step, replicate]
        ).fetchall()

        if not result:
            raise ValueError(
                f"No spatial data found for config_hash={config_hash[:8]}..., "
                f"step={step}, replicate={replicate}."
            )

        lons = [row[0] for row in result]
        lats = [row[1] for row in result]
        values = [row[2] for row in result]

        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(lons, lats, c=values, cmap="viridis", alpha=0.8)
        fig.colorbar(scatter, ax=ax, label=variable)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{variable} at step {step}")

        ax.set_aspect("equal")
        fig.tight_layout()

        if show:
            plt.show()

        return fig
