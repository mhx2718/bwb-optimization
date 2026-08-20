"""Authoritative dataset schema and unit conventions.

The structural forward model deliberately excludes the three flight-condition
columns.  The stress classifier uses all 24 inputs because stress feasibility
can depend on the operating point.
"""

from __future__ import annotations

from collections.abc import Iterable


GEOMETRY_COLUMNS = (
    "C2/C1",
    "C3/C1",
    "C4/C1",
    "B1/C1",
    "B2/C1",
    "B3/C1",
    "X3/C1",
    "S1",
    "S3",
    "C1",
)

STRUCTURE_COLUMNS = (
    "Skin Thickness",
    "Front Spar Chord %",
    "Rear Spar Chord %",
    "Spar Thickness",
    "# of Ribs",
    "Rib Thickness",
    "Wingbox Cutout",
    "# of Fuselage Ribs",
    "# of Fuselage Spars",
    "Fuselage Struct Thickness",
    "Fuselage Struct Width",
)

DESIGN_COLUMNS = GEOMETRY_COLUMNS + STRUCTURE_COLUMNS
FLIGHT_COLUMNS = ("Altitude", "KCAS", "AOA")
ALL_INPUT_COLUMNS = DESIGN_COLUMNS + FLIGHT_COLUMNS

FORWARD_TARGET_COLUMNS = (
    "Aircraft Empty Weight",
    "Payload Volume",
    "Fuel Volume",
)
STRESS_TARGET_COLUMN = "Max Hotspot Stress"
ALL_TARGET_COLUMNS = FORWARD_TARGET_COLUMNS + (STRESS_TARGET_COLUMN,)
ALL_COLUMNS = ALL_INPUT_COLUMNS + ALL_TARGET_COLUMNS

TOPOLOGY_COLUMNS = (
    "# of Ribs",
    "# of Fuselage Ribs",
    "# of Fuselage Spars",
)
CONTINUOUS_DESIGN_COLUMNS = tuple(
    name for name in DESIGN_COLUMNS if name not in TOPOLOGY_COLUMNS
)

# One authoritative source for optimization, perturbation, plotting, and k-NN
# scaling.  Keeping this map beside the ordered schema prevents silent drift
# between evaluators and diagnostics.
OFFICIAL_DESIGN_BOUNDS = {
    "C2/C1": (0.55, 0.85),
    "C3/C1": (0.18, 0.28),
    "C4/C1": (0.06, 0.09),
    "B1/C1": (0.10, 0.20),
    "B2/C1": (0.05, 0.20),
    "B3/C1": (0.35, 0.70),
    "X3/C1": (0.50, 0.65),
    "S1": (40.0, 60.0),
    "S3": (20.0, 40.0),
    "C1": (2500.0, 4000.0),
    "Skin Thickness": (0.0003, 0.0050),
    "Front Spar Chord %": (0.18, 0.35),
    "Rear Spar Chord %": (0.55, 0.75),
    "Spar Thickness": (0.00098, 0.0080),
    "# of Ribs": (3.0, 14.0),
    "Rib Thickness": (0.0015, 0.0150),
    "Wingbox Cutout": (0.01, 0.05),
    "# of Fuselage Ribs": (3.0, 11.0),
    "# of Fuselage Spars": (3.0, 12.0),
    "Fuselage Struct Thickness": (0.002, 0.025),
    "Fuselage Struct Width": (0.0010, 0.0150),
}

VOLUME_TO_CUBIC_METERS = 1.0e-9
FORWARD_TARGET_SCALES_TO_SI = (1.0, VOLUME_TO_CUBIC_METERS, VOLUME_TO_CUBIC_METERS)
STRESS_LIMIT_MPA = 335.0


def canonicalize_column_names(columns: Iterable[object]) -> tuple[str, ...]:
    """Strip accidental surrounding whitespace and reject ambiguous names."""

    canonical = tuple(str(column).strip() for column in columns)
    if len(set(canonical)) != len(canonical):
        duplicates = sorted(
            {name for name in canonical if canonical.count(name) > 1}
        )
        raise ValueError(
            "Column-name normalization created duplicates: "
            f"{duplicates}."
        )
    return canonical


def assert_exact_schema(columns: Iterable[object]) -> None:
    """Require exactly the documented 28 columns, independent of input order."""

    actual = canonicalize_column_names(columns)
    missing = sorted(set(ALL_COLUMNS) - set(actual))
    unexpected = sorted(set(actual) - set(ALL_COLUMNS))
    if missing or unexpected or len(actual) != len(ALL_COLUMNS):
        raise ValueError(
            "Dataset schema mismatch. "
            f"Missing={missing}; unexpected={unexpected}; "
            f"expected {len(ALL_COLUMNS)} columns, received {len(actual)}."
        )


if len(DESIGN_COLUMNS) != 21 or len(ALL_INPUT_COLUMNS) != 24:
    raise RuntimeError("The BWB schema must contain 21 design and 24 total inputs.")
if len(CONTINUOUS_DESIGN_COLUMNS) != 18:
    raise RuntimeError("The BWB schema must contain 18 continuous design inputs.")
if tuple(OFFICIAL_DESIGN_BOUNDS) != DESIGN_COLUMNS:
    raise RuntimeError("Official bounds must preserve the exact 21D design order.")
