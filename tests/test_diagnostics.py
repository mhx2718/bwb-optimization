import numpy as np
import pandas as pd
import pytest

from bwb_pipeline.diagnostics import (
    DESIGN_COLUMNS,
    OFFICIAL_DESIGN_BOUNDS,
    compute_loo_knn_diagnostics,
    compute_query_knn_support,
    deduplicate_designs,
    empirical_pareto_archive,
)


def full_design(**updates):
    values = {
        column: 0.5 * (OFFICIAL_DESIGN_BOUNDS[column][0] + upper)
        for column, (_, upper) in OFFICIAL_DESIGN_BOUNDS.items()
    }
    values.update(updates)
    return values


def test_loo_deduplicates_21d_designs_before_distance():
    first = full_design(**{"C2/C1": 0.55})
    second = full_design(**{"C2/C1": 0.65})
    third = full_design(**{"C2/C1": 0.85})
    dataset = pd.DataFrame(
        [first, first, second, third],
        index=["same-flight-a", "same-flight-b", "second", "third"],
    )
    dataset["Altitude"] = [5.0, 15.0, 5.0, 15.0]

    diagnostics = compute_loo_knn_diagnostics(dataset, k=1)
    unique = diagnostics.unique_designs
    rows = diagnostics.dataset_rows

    assert len(unique) == 3
    assert sorted(unique["source_row_count"].tolist()) == [1, 1, 2]
    duplicate_rows = rows.iloc[:2]
    assert duplicate_rows["design_id"].nunique() == 1
    assert duplicate_rows["loo_nn1_distance"].nunique() == 1
    # The repeated flight row is excluded as the same design, so its distance
    # is positive instead of the misleading zero from row-wise LOO.
    assert duplicate_rows["loo_nn1_distance"].iloc[0] > 0.0


def test_simple_loo_distances_are_exact_and_row_order_independent():
    table = pd.DataFrame({"x": [0.0, 1.0, 3.0], "y": [0.0, 0.0, 0.0]})
    bounds = {"x": (0.0, 3.0), "y": (0.0, 1.0)}

    first = compute_loo_knn_diagnostics(
        table,
        k=1,
        design_columns=("x", "y"),
        bounds=bounds,
    ).unique_designs.sort_values("design_id")
    second = compute_loo_knn_diagnostics(
        table.iloc[[2, 0, 1]].reset_index(drop=True),
        k=1,
        design_columns=("x", "y"),
        bounds=bounds,
    ).unique_designs.sort_values("design_id")

    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )
    by_x = first.set_index("x")
    assert by_x.loc[0.0, "loo_nn1_distance"] == pytest.approx(1.0 / 3.0)
    assert by_x.loc[1.0, "loo_nn1_distance"] == pytest.approx(1.0 / 3.0)
    assert by_x.loc[3.0, "loo_nn1_distance"] == pytest.approx(2.0 / 3.0)


def test_query_support_reports_percentile_and_seen_topology():
    reference = pd.DataFrame(
        [
            full_design(**{"C2/C1": value, "# of Ribs": ribs})
            for value, ribs in [
                (0.55, 4),
                (0.60, 4),
                (0.65, 5),
                (0.75, 5),
                (0.85, 6),
                (0.80, 6),
            ]
        ]
    )
    query = pd.DataFrame(
        [
            full_design(**{"C2/C1": 0.61, "# of Ribs": 4}),
            full_design(**{"C2/C1": 0.61, "# of Ribs": 14}),
        ],
        index=["seen", "unseen"],
    )
    support = compute_query_knn_support(reference, query, k=2)

    assert support["query_source_index"].tolist() == ["seen", "unseen"]
    assert support["topology_seen"].tolist() == [True, False]
    assert support["reference_nn1_distance"].ge(0.0).all()
    assert support["reference_loo_percentile"].between(0.0, 100.0).all()
    assert set(support["support_band"]).issubset(
        {"within_loo_q95", "between_loo_q95_q99", "beyond_loo_q99"}
    )


def test_empirical_pareto_filters_hard_failures_and_dominated_points():
    candidates = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "f1": [0.0, 1.0, 2.0, 2.0, -1.0, 0.0],
            "f2": [3.0, 2.0, 1.0, 3.0, 0.0, 3.0],
            "hard_accepted": [True, True, True, True, False, True],
            "official_loss": [1.5, 1.4, 1.3, 2.5, 0.0, 1.5],
        }
    )
    front = empirical_pareto_archive(
        candidates,
        objective_columns=("f1", "f2"),
        design_columns=("x", "y"),
    )

    assert len(front) == 3
    assert set(front["x"]) == {0.0, 1.0, 2.0}
    assert front["hard_accepted_normalized"].all()
    assert front["finite_objectives"].all()
    assert front["is_empirical_pareto"].all()
    assert front["official_loss"].tolist() == sorted(front["official_loss"])


def test_empirical_pareto_excludes_nonfinite_objectives():
    candidates = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "f1": [np.nan, 1.0],
            "f2": [0.0, 1.0],
            "hard_accepted": [True, True],
            "official_loss": [0.0, 1.0],
        }
    )
    front = empirical_pareto_archive(
        candidates,
        objective_columns=("f1", "f2"),
        design_columns=("x", "y"),
    )
    assert front["x"].tolist() == [1.0]


def test_inconsistent_duplicate_common_evaluation_raises():
    candidates = pd.DataFrame(
        {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "f1": [1.0, 1.1],
            "f2": [2.0, 2.0],
            "hard_accepted": [True, True],
            "official_loss": [1.0, 1.1],
        }
    )
    with pytest.raises(ValueError, match="inconsistent"):
        empirical_pareto_archive(
            candidates,
            objective_columns=("f1", "f2"),
            design_columns=("x", "y"),
        )


def test_loo_requires_more_unique_designs_than_k():
    table = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]})
    with pytest.raises(ValueError, match="requires at least"):
        compute_loo_knn_diagnostics(
            table,
            k=2,
            design_columns=("x", "y"),
            bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        )


def test_deduplication_rejects_nonfinite_design_values():
    table = pd.DataFrame({column: [0.0] for column in DESIGN_COLUMNS})
    table.loc[0, DESIGN_COLUMNS[0]] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        deduplicate_designs(table)
