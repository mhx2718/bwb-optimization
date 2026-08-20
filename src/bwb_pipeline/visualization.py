"""Deterministic, publication-ready optimization diagnostics.

The plotting helpers do not draw random jitter or use stochastic KDEs.  Any
dataset downsampling is based on a stable content hash and an explicit seed,
and the exact sampled rows are returned to the caller for optional export.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .diagnostics import (
    DEFAULT_OBJECTIVE_COLUMNS,
    DESIGN_COLUMNS,
    OFFICIAL_DESIGN_BOUNDS,
    deduplicate_designs,
)


PAIRPLOT_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "geometry_chord_span": (
        "C2/C1",
        "C3/C1",
        "C4/C1",
        "B1/C1",
        "B2/C1",
        "B3/C1",
    ),
    "geometry_position_angles_scale": (
        "X3/C1",
        "S1",
        "S3",
        "C1",
    ),
    "wing_structure": (
        "Skin Thickness",
        "Front Spar Chord %",
        "Rear Spar Chord %",
        "Spar Thickness",
        "# of Ribs",
        "Rib Thickness",
    ),
    "cutout_fuselage_structure": (
        "Wingbox Cutout",
        "# of Fuselage Ribs",
        "# of Fuselage Spars",
        "Fuselage Struct Thickness",
        "Fuselage Struct Width",
    ),
}


@dataclass(frozen=True)
class PairplotData:
    """Exact plot inputs, suitable for export alongside a figure."""

    dataset_unique: pd.DataFrame
    dataset_scatter: pd.DataFrame
    generated: pd.DataFrame
    selected: pd.DataFrame


def _require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise KeyError(f"Missing plotting columns: {missing}")


def _stable_sample(
    table: pd.DataFrame,
    maximum_rows: int,
    *,
    seed: int,
    id_column: str = "design_id",
) -> pd.DataFrame:
    maximum_rows = int(maximum_rows)
    if maximum_rows < 1:
        raise ValueError("maximum_rows must be positive.")
    if len(table) <= maximum_rows:
        return table.copy().reset_index(drop=True)
    _require_columns(table, [id_column])
    seed_text = str(int(seed)).encode("ascii") + b":"
    priorities = [
        hashlib.sha256(seed_text + str(value).encode("utf-8")).hexdigest()
        for value in table[id_column]
    ]
    sampled = table.assign(_sample_priority=priorities).sort_values(
        ["_sample_priority", id_column], kind="stable"
    )
    return sampled.head(maximum_rows).drop(columns="_sample_priority").reset_index(
        drop=True
    )


def _attach_content_ids(
    table: pd.DataFrame,
    design_columns: Sequence[str],
) -> pd.DataFrame:
    _require_columns(table, design_columns)
    result = table.copy().reset_index(drop=True)
    values = result.loc[:, design_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Plot design columns must be finite.")
    prefix = ("\x1f".join(design_columns) + "\x1e").encode("utf-8")
    result["design_id"] = [
        hashlib.sha256(
            prefix + np.asarray(row, dtype="<f8").tobytes()
        ).hexdigest()
        for row in values
    ]
    return result


def prepare_pairplot_data(
    dataset: pd.DataFrame,
    generated: pd.DataFrame,
    selected: Optional[pd.DataFrame] = None,
    *,
    design_columns: Sequence[str] = DESIGN_COLUMNS,
    maximum_dataset_rows: int = 3000,
    sample_seed: int = 0,
) -> PairplotData:
    """Prepare unique and deterministically sampled pairplot inputs."""

    dataset_unique, _ = deduplicate_designs(
        dataset, design_columns=design_columns
    )
    dataset_scatter = _stable_sample(
        dataset_unique,
        maximum_dataset_rows,
        seed=sample_seed,
    )
    generated_with_ids = _attach_content_ids(generated, design_columns)
    selected_frame = (
        generated.iloc[0:0].copy()
        if selected is None
        else selected.copy()
    )
    selected_with_ids = _attach_content_ids(selected_frame, design_columns)
    return PairplotData(
        dataset_unique=dataset_unique,
        dataset_scatter=dataset_scatter,
        generated=generated_with_ids,
        selected=selected_with_ids,
    )


def _axis_limits(*arrays: np.ndarray) -> Tuple[float, float]:
    finite = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        finite.append(values[np.isfinite(values)])
    nonempty = [values for values in finite if len(values)]
    if not nonempty:
        return -1.0, 1.0
    values = np.concatenate(nonempty)
    lower, upper = float(values.min()), float(values.max())
    span = upper - lower
    padding = 0.04 * span if span > 0.0 else 0.04 * max(abs(lower), 1.0)
    return lower - padding, upper + padding


def _case_colors(values: pd.Series) -> Dict[object, Tuple[float, ...]]:
    cases = sorted(values.dropna().unique().tolist(), key=str)
    cmap = plt.get_cmap("tab10")
    return {case: cmap(index % 10) for index, case in enumerate(cases)}


def plot_design_pairplot(
    data: PairplotData,
    feature_names: Sequence[str],
    *,
    case_column: str = "case_id",
    bounds: Mapping[str, Sequence[float]] = OFFICIAL_DESIGN_BOUNDS,
    title: Optional[str] = None,
) -> Figure:
    """Plot dataset support, accepted generated designs, and final selections."""

    features = tuple(feature_names)
    if len(features) < 2:
        raise ValueError("A pairplot requires at least two features.")
    for frame in (
        data.dataset_unique,
        data.dataset_scatter,
        data.generated,
        data.selected,
    ):
        _require_columns(frame, features)
    missing_bounds = [feature for feature in features if feature not in bounds]
    if missing_bounds:
        raise KeyError(f"Missing pairplot bounds: {missing_bounds}")

    if case_column in data.generated:
        colors = _case_colors(data.generated[case_column])
    else:
        colors = {"generated": plt.get_cmap("tab10")(0)}

    n_features = len(features)
    if data.dataset_unique.empty:
        raise ValueError("The unique dataset pairplot reference is empty.")
    figure, axes = plt.subplots(
        n_features,
        n_features,
        figsize=(2.35 * n_features, 2.35 * n_features),
        squeeze=False,
    )
    for row_index, y_name in enumerate(features):
        dataset_y = data.dataset_unique[y_name].to_numpy(dtype=float)
        y_q01, y_q99 = np.quantile(dataset_y, [0.01, 0.99])
        for column_index, x_name in enumerate(features):
            axis = axes[row_index, column_index]
            dataset_x = data.dataset_unique[x_name].to_numpy(dtype=float)
            generated_x = data.generated[x_name].to_numpy(dtype=float)
            selected_x = data.selected[x_name].to_numpy(dtype=float)
            x_q01, x_q99 = np.quantile(dataset_x, [0.01, 0.99])

            if row_index == column_index:
                combined = np.concatenate(
                    [dataset_x, generated_x, selected_x]
                )
                edges = np.histogram_bin_edges(combined, bins=24)
                axis.hist(
                    dataset_x,
                    bins=edges,
                    density=True,
                    color="0.72",
                    alpha=0.65,
                    edgecolor="none",
                )
                if case_column in data.generated:
                    for case, color in colors.items():
                        values = data.generated.loc[
                            data.generated[case_column] == case, x_name
                        ].to_numpy(dtype=float)
                        if len(values):
                            axis.hist(
                                values,
                                bins=edges,
                                density=True,
                                histtype="step",
                                linewidth=1.3,
                                color=color,
                            )
                elif len(generated_x):
                    axis.hist(
                        generated_x,
                        bins=edges,
                        density=True,
                        histtype="step",
                        linewidth=1.3,
                        color=colors["generated"],
                    )
                axis.axvline(x_q01, color="0.35", linestyle="--", linewidth=0.8)
                axis.axvline(x_q99, color="0.35", linestyle="--", linewidth=0.8)
                lower, upper = bounds[x_name]
                axis.axvline(lower, color="black", linestyle=":", linewidth=0.8)
                axis.axvline(upper, color="black", linestyle=":", linewidth=0.8)
                if len(data.selected):
                    y_level = 0.90 * axis.get_ylim()[1]
                    for _, selected in data.selected.iterrows():
                        case = selected.get(case_column, "generated")
                        axis.scatter(
                            selected[x_name],
                            y_level,
                            marker="*",
                            s=75,
                            color=colors.get(case, colors.get("generated", "crimson")),
                            edgecolor="black",
                            linewidth=0.55,
                            zorder=5,
                        )
                axis.set_title(x_name, fontsize=9)
                axis.set_xlim(
                    _axis_limits(
                        dataset_x,
                        generated_x,
                        selected_x,
                        np.asarray(bounds[x_name], dtype=float),
                    )
                )
            else:
                generated_y = data.generated[y_name].to_numpy(dtype=float)
                selected_y = data.selected[y_name].to_numpy(dtype=float)
                axis.add_patch(
                    Rectangle(
                        (x_q01, y_q01),
                        x_q99 - x_q01,
                        y_q99 - y_q01,
                        facecolor="tab:blue",
                        alpha=0.035,
                        edgecolor="0.25",
                        linestyle="--",
                        linewidth=0.8,
                        zorder=0,
                    )
                )
                axis.scatter(
                    data.dataset_scatter[x_name],
                    data.dataset_scatter[y_name],
                    s=7,
                    color="0.58",
                    alpha=0.20,
                    linewidths=0,
                    rasterized=True,
                )
                if case_column in data.generated:
                    for case, color in colors.items():
                        rows = data.generated.loc[
                            data.generated[case_column] == case
                        ]
                        axis.scatter(
                            rows[x_name],
                            rows[y_name],
                            s=16,
                            color=color,
                            alpha=0.55,
                            linewidths=0,
                        )
                elif len(data.generated):
                    axis.scatter(
                        generated_x,
                        generated_y,
                        s=16,
                        color=colors["generated"],
                        alpha=0.55,
                        linewidths=0,
                    )
                for _, selected in data.selected.iterrows():
                    case = selected.get(case_column, "generated")
                    axis.scatter(
                        selected[x_name],
                        selected[y_name],
                        marker="*",
                        s=95,
                        color=colors.get(case, colors.get("generated", "crimson")),
                        edgecolor="black",
                        linewidth=0.6,
                        zorder=5,
                    )
                axis.set_xlim(
                    _axis_limits(
                        dataset_x,
                        generated_x,
                        selected_x,
                        np.asarray(bounds[x_name], dtype=float),
                    )
                )
                axis.set_ylim(
                    _axis_limits(
                        dataset_y,
                        generated_y,
                        selected_y,
                        np.asarray(bounds[y_name], dtype=float),
                    )
                )

            axis.grid(alpha=0.12, linewidth=0.6)
            axis.tick_params(axis="both", labelsize=7)
            if row_index < n_features - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(x_name, fontsize=8)
                axis.tick_params(axis="x", labelrotation=30)
            if column_index > 0:
                axis.tick_params(labelleft=False)
            elif row_index != column_index:
                axis.set_ylabel(y_name, fontsize=8)

    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=5,
            color="0.58",
            label="Unique dataset designs",
        ),
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.35",
            label="Dataset 1st–99th percentiles",
        ),
        Line2D(
            [],
            [],
            linestyle=":",
            color="black",
            label="Official bounds",
        ),
    ]
    for case, color in colors.items():
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=5,
                color=color,
                label=f"Generated {case}",
            )
        )
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="*",
            markersize=10,
            markerfacecolor="white",
            markeredgecolor="black",
            label="Final selected designs",
        )
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=min(4, len(handles)),
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(title or "Generated designs relative to dataset support", y=1.018)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return figure


def plot_grouped_design_pairplots(
    data: PairplotData,
    *,
    groups: Mapping[str, Sequence[str]] = PAIRPLOT_GROUPS,
    case_column: str = "case_id",
    bounds: Mapping[str, Sequence[float]] = OFFICIAL_DESIGN_BOUNDS,
) -> Dict[str, Figure]:
    """Create the four readable pairplots covering all 21 variables."""

    flattened = [feature for features in groups.values() for feature in features]
    missing = sorted(set(DESIGN_COLUMNS) - set(flattened))
    duplicates = sorted(
        feature for feature in set(flattened) if flattened.count(feature) > 1
    )
    if missing or duplicates:
        raise ValueError(
            f"Invalid pairplot groups: missing={missing}, duplicates={duplicates}"
        )
    return {
        group_name: plot_design_pairplot(
            data,
            features,
            case_column=case_column,
            bounds=bounds,
            title=f"Generated designs: {group_name}",
        )
        for group_name, features in groups.items()
    }


def plot_empirical_pareto(
    pareto: pd.DataFrame,
    *,
    objective_columns: Sequence[str] = DEFAULT_OBJECTIVE_COLUMNS,
    score_column: str = "official_loss",
    selected: Optional[pd.DataFrame] = None,
) -> Figure:
    """Plot every pairwise projection of the empirical nondominated archive."""

    objectives = tuple(objective_columns)
    if len(objectives) < 2:
        raise ValueError("At least two objectives are required.")
    _require_columns(pareto, list(objectives) + [score_column])
    if pareto.empty:
        raise ValueError("The empirical Pareto archive is empty.")
    if selected is not None and len(selected):
        _require_columns(selected, objectives)
    values = pareto.loc[:, objectives].to_numpy(dtype=float)
    scores = pareto[score_column].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(scores).all():
        raise ValueError("Pareto plotting values must be finite.")

    n_objectives = len(objectives)
    figure, axes = plt.subplots(
        n_objectives,
        n_objectives,
        figsize=(2.55 * n_objectives, 2.55 * n_objectives),
        squeeze=False,
    )
    scatter_for_colorbar = None
    for row_index, y_name in enumerate(objectives):
        for column_index, x_name in enumerate(objectives):
            axis = axes[row_index, column_index]
            if row_index == column_index:
                axis.hist(
                    pareto[x_name],
                    bins=20,
                    color="0.55",
                    alpha=0.65,
                    edgecolor="none",
                )
                axis.set_title(x_name, fontsize=8)
            else:
                scatter_for_colorbar = axis.scatter(
                    pareto[x_name],
                    pareto[y_name],
                    c=scores,
                    cmap="viridis_r",
                    s=24,
                    alpha=0.80,
                    linewidths=0,
                )
                if selected is not None and len(selected):
                    axis.scatter(
                        selected[x_name],
                        selected[y_name],
                        marker="*",
                        s=130,
                        color="crimson",
                        edgecolor="black",
                        linewidth=0.6,
                        zorder=4,
                    )
            axis.grid(alpha=0.15, linewidth=0.6)
            axis.tick_params(labelsize=7)
            if row_index == n_objectives - 1:
                axis.set_xlabel(x_name, fontsize=8)
            else:
                axis.tick_params(labelbottom=False)
            if column_index == 0 and row_index != column_index:
                axis.set_ylabel(y_name, fontsize=8)
            elif column_index > 0:
                axis.tick_params(labelleft=False)
    if scatter_for_colorbar is not None:
        figure.colorbar(
            scatter_for_colorbar,
            ax=axes.ravel().tolist(),
            label="Official weighted loss",
            shrink=0.78,
        )
    figure.suptitle("Empirical nondominated archive", y=1.005)
    figure.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.08,
        top=0.94,
        wspace=0.14,
        hspace=0.14,
    )
    return figure


def plot_knn_support_ecdf(
    reference_loo: pd.DataFrame,
    optimized_support: pd.DataFrame,
    *,
    loo_column: str = "loo_knn_mean_distance",
    optimized_column: str = "reference_knn_mean_distance",
    label_column: str = "case_id",
) -> Figure:
    """Compare optimized-design k-NN distances with the dataset LOO ECDF."""

    _require_columns(reference_loo, [loo_column])
    _require_columns(optimized_support, [optimized_column])
    values = np.sort(reference_loo[loo_column].to_numpy(dtype=float))
    optimized = optimized_support[optimized_column].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(optimized).all():
        raise ValueError("k-NN plotting values must be finite.")
    if len(values) == 0:
        raise ValueError("Reference LOO table is empty.")

    ecdf = np.arange(1, len(values) + 1, dtype=float) / len(values)
    figure, axis = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    axis.step(values, ecdf, where="post", color="0.2", linewidth=1.6)
    q95, q99 = np.quantile(values, [0.95, 0.99])
    axis.axvline(q95, color="tab:orange", linestyle="--", label="LOO q95")
    axis.axvline(q99, color="tab:red", linestyle="--", label="LOO q99")
    for position, value in enumerate(optimized):
        percentile = float(np.mean(values <= value))
        if label_column in optimized_support:
            label = str(optimized_support.iloc[position][label_column])
        else:
            label = f"optimized {position + 1}"
        axis.scatter(
            value,
            percentile,
            marker="*",
            s=110,
            edgecolor="black",
            linewidth=0.5,
            label=label,
            zorder=4,
        )
    axis.set_xlabel("Official-bound-scaled mean k-NN distance")
    axis.set_ylabel("Dataset LOO empirical CDF")
    axis.set_title("Optimized designs relative to dataset support")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    return figure
