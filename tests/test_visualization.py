import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd

from bwb_pipeline.diagnostics import OFFICIAL_DESIGN_BOUNDS
from bwb_pipeline.visualization import (
    plot_design_pairplot,
    plot_empirical_pareto,
    plot_knn_support_ecdf,
    prepare_pairplot_data,
)


def design(offset=0.0, case_id=None):
    row = {
        column: 0.5 * (lower + upper)
        for column, (lower, upper) in OFFICIAL_DESIGN_BOUNDS.items()
    }
    row["C2/C1"] += offset
    row["B1/C1"] += 0.2 * offset
    if case_id is not None:
        row["case_id"] = case_id
    return row


def test_pairplot_subsample_is_content_deterministic():
    dataset = pd.DataFrame([design(0.005 * index) for index in range(10)])
    generated = pd.DataFrame([design(0.010, 1), design(0.020, 2)])
    selected = generated.iloc[[0]].copy()

    first = prepare_pairplot_data(
        dataset,
        generated,
        selected,
        maximum_dataset_rows=5,
        sample_seed=17,
    )
    second = prepare_pairplot_data(
        dataset.iloc[::-1],
        generated,
        selected,
        maximum_dataset_rows=5,
        sample_seed=17,
    )
    assert first.dataset_scatter["design_id"].tolist() == second.dataset_scatter[
        "design_id"
    ].tolist()

    figure = plot_design_pairplot(
        first,
        ["C2/C1", "B1/C1"],
        title="test",
    )
    assert len(figure.axes) == 4
    plt.close(figure)


def test_pareto_and_knn_plots_render_without_stochastic_state():
    pareto = pd.DataFrame(
        {
            "f1": [0.0, 0.5, 1.0],
            "f2": [1.0, 0.5, 0.0],
            "official_loss": [0.6, 0.5, 0.4],
        }
    )
    selected = pareto.iloc[[2]]
    pareto_figure = plot_empirical_pareto(
        pareto,
        objective_columns=("f1", "f2"),
        selected=selected,
    )
    assert len(pareto_figure.axes) >= 4
    plt.close(pareto_figure)

    loo = pd.DataFrame({"loo_knn_mean_distance": [0.1, 0.2, 0.3, 0.4]})
    optimized = pd.DataFrame(
        {"reference_knn_mean_distance": [0.25], "case_id": [1]}
    )
    knn_figure = plot_knn_support_ecdf(loo, optimized)
    assert len(knn_figure.axes) == 1
    plt.close(knn_figure)
