"""Deterministic, differentiable 21-D structural forward surrogate."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import (
    ArtifactManifest,
    ArtifactIntegrityError,
    assert_artifact_compatible,
    assert_artifact_payloads,
    atomic_output_path,
    atomic_write_json,
    installed_versions,
    load_artifact_manifest,
    require_complete_artifact,
    save_artifact_manifest,
    source_fingerprint,
)
from ..config import ForwardModelConfig
from ..data import DataBundle, SPLIT_NAMES
from ..metrics import regression_metrics
from ..reproducibility import SeedRegistry, set_global_determinism
from ..schema import (
    DESIGN_COLUMNS,
    FORWARD_TARGET_COLUMNS,
    FORWARD_TARGET_SCALES_TO_SI,
)

try:
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    torch = None
    nn = None
    torch_functional = None
    DataLoader = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


FORWARD_ARTIFACT_PACKAGES = ("numpy", "pandas", "scikit-learn", "torch")


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for the forward surrogate. Install the locked "
            "project dependencies before training or loading this model."
        ) from _TORCH_IMPORT_ERROR


if nn is not None:

    class _ResidualBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.linear_1 = nn.Linear(width, width)
            self.linear_2 = nn.Linear(width, width)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            hidden = self.norm(values)
            hidden = torch_functional.silu(self.linear_1(hidden))
            hidden = self.linear_2(hidden)
            return values + hidden


    class ResidualForwardNet(nn.Module):
        """Shared residual backbone with three positive structural outputs."""

        def __init__(
            self,
            input_dim: int = len(DESIGN_COLUMNS),
            output_dim: int = len(FORWARD_TARGET_COLUMNS),
            hidden_width: int = 384,
            residual_blocks: int = 4,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.output_dim = int(output_dim)
            self.hidden_width = int(hidden_width)
            self.residual_blocks = int(residual_blocks)
            self.stem = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_width),
                nn.LayerNorm(self.hidden_width),
                nn.SiLU(),
            )
            self.blocks = nn.ModuleList(
                _ResidualBlock(self.hidden_width)
                for _ in range(self.residual_blocks)
            )
            self.final_norm = nn.LayerNorm(self.hidden_width)
            self.heads = nn.ModuleList(
                nn.Linear(self.hidden_width, 1) for _ in range(self.output_dim)
            )
            # Normalized positive targets are centered near one.
            inverse_softplus_one = float(np.log(np.expm1(1.0)))
            for head in self.heads:
                nn.init.constant_(head.bias, inverse_softplus_one)

        def forward(self, standardized_designs: torch.Tensor) -> torch.Tensor:
            hidden = self.stem(standardized_designs)
            for block in self.blocks:
                hidden = block(hidden)
            hidden = torch_functional.silu(self.final_norm(hidden))
            raw = torch.cat([head(hidden) for head in self.heads], dim=-1)
            return torch_functional.softplus(raw) + 1.0e-8

else:

    class ResidualForwardNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


@dataclass(frozen=True)
class ForwardScalers:
    input_mean: np.ndarray
    input_scale: np.ndarray
    target_positive_mean: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "input_mean": self.input_mean.tolist(),
            "input_scale": self.input_scale.tolist(),
            "target_positive_mean": self.target_positive_mean.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ForwardScalers":
        return cls(
            input_mean=np.asarray(values["input_mean"], dtype=np.float64),
            input_scale=np.asarray(values["input_scale"], dtype=np.float64),
            target_positive_mean=np.asarray(
                values["target_positive_mean"], dtype=np.float64
            ),
        )


class ForwardPredictor:
    """Validated NumPy and differentiable Tensor interfaces in physical units."""

    feature_names = DESIGN_COLUMNS
    target_names = FORWARD_TARGET_COLUMNS

    def __init__(
        self,
        model: Any,
        scalers: ForwardScalers,
        *,
        device: str = "cpu",
        artifact_id: str | None = None,
    ) -> None:
        _require_torch()
        self.model = model.to(device)
        self.model.eval()
        self.scalers = scalers
        self.device = str(device)
        self.artifact_id = artifact_id

    def _array(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            missing = [name for name in self.feature_names if name not in values.columns]
            if missing:
                raise ValueError(f"Missing forward-model columns: {missing}.")
            array = values.loc[:, self.feature_names].to_numpy(dtype=np.float64)
        else:
            array = np.asarray(values, dtype=np.float64)
            if array.ndim == 1:
                array = array[None, :]
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected forward input shape (N, {len(self.feature_names)}), "
                f"received {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("Forward-model inputs must be finite.")
        return array

    def predict(
        self,
        values: pd.DataFrame | np.ndarray,
        *,
        batch_size: int = 8192,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        array = self._array(values)
        if len(array) == 0:
            return np.empty((0, len(self.target_names)), dtype=np.float64)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(array), batch_size):
                batch = torch.as_tensor(
                    array[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                outputs.append(self.predict_tensor(batch).detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float64, copy=False)

    def predict_tensor(
        self,
        values: Any,
        *,
        input_is_scaled: bool = False,
    ) -> Any:
        """Return physical-unit predictions while preserving autograd."""

        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected forward tensor shape (N, {len(self.feature_names)}), "
                f"received {tuple(values.shape)}."
            )
        values = values.to(device=self.device, dtype=torch.float32)
        if input_is_scaled:
            standardized = values
        else:
            mean = torch.as_tensor(
                self.scalers.input_mean, dtype=values.dtype, device=values.device
            )
            scale = torch.as_tensor(
                self.scalers.input_scale, dtype=values.dtype, device=values.device
            )
            standardized = (values - mean) / scale
        normalized = self.model(standardized)
        target_scale = torch.as_tensor(
            self.scalers.target_positive_mean,
            dtype=normalized.dtype,
            device=normalized.device,
        )
        return normalized * target_scale


def _training_config(config: ForwardModelConfig) -> dict[str, Any]:
    values = asdict(config)
    values.pop("output_dir", None)
    values.pop("force_retrain", None)
    return values


def _expected_manifest(
    bundle: DataBundle,
    config: ForwardModelConfig,
    seed_registry: SeedRegistry,
) -> ArtifactManifest:
    module_root = Path(__file__).resolve().parents[1]
    code_hash = source_fingerprint(
        (
            Path(__file__),
            module_root / "schema.py",
            module_root / "config.py",
            module_root / "data.py",
            module_root / "metrics.py",
            module_root / "reproducibility.py",
            module_root / "artifacts.py",
        )
    )
    return ArtifactManifest(
        artifact_type="bwb-structural-forward-21d-v1",
        dataset_fingerprint=bundle.dataset_fingerprint,
        split_fingerprint=bundle.split_manifest.split_fingerprint,
        feature_names=DESIGN_COLUMNS,
        target_names=FORWARD_TARGET_COLUMNS,
        master_seed=seed_registry.master_seed,
        model_config=_training_config(config),
        unit_transforms={
            "target_scaling": "positive_train_mean",
            "output_scales_to_SI": FORWARD_TARGET_SCALES_TO_SI,
            "output_units_after_scaling": ("kg", "m^3", "m^3"),
        },
        package_versions=installed_versions(FORWARD_ARTIFACT_PACKAGES),
        code_version=f"0.1.0+{code_hash}",
    ).with_artifact_id()


def _make_model(config: ForwardModelConfig) -> Any:
    _require_torch()
    return ResidualForwardNet(
        input_dim=len(DESIGN_COLUMNS),
        output_dim=len(FORWARD_TARGET_COLUMNS),
        hidden_width=config.hidden_width,
        residual_blocks=config.residual_blocks,
    )


def _load_predictor(
    checkpoint_path: Path,
    config: ForwardModelConfig,
    device: str,
    artifact_id: str,
) -> ForwardPredictor:
    _require_torch()
    payload = torch.load(checkpoint_path, map_location=device)
    if payload.get("artifact_id") != artifact_id:
        raise ArtifactIntegrityError(
            "Forward checkpoint artifact_id does not match its verified manifest."
        )
    model = _make_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    scalers = ForwardScalers.from_dict(payload["scalers"])
    return ForwardPredictor(
        model, scalers, device=device, artifact_id=artifact_id
    )


def _predict_array(
    model: Any,
    input_values: np.ndarray,
    scalers: ForwardScalers,
    device: str,
) -> np.ndarray:
    predictor = ForwardPredictor(model, scalers, device=device)
    return predictor.predict(input_values)


def train_or_load_forward(
    bundle: DataBundle,
    config: ForwardModelConfig,
    seed_registry: SeedRegistry,
    *,
    artifact_dir: str | Path | None = None,
    device: str = "cpu",
) -> ForwardPredictor:
    """Load an exact-compatible artifact or train a new deterministic model."""

    _require_torch()
    output_dir = Path(artifact_dir) if artifact_dir is not None else config.output_dir
    checkpoint_path = output_dir / "forward_model.pt"
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "history.csv"
    test_predictions_path = output_dir / "test_predictions.csv"
    expected = _expected_manifest(bundle, config, seed_registry)

    if not config.force_retrain and require_complete_artifact(
        (checkpoint_path, manifest_path)
    ):
        actual = load_artifact_manifest(manifest_path)
        assert_artifact_compatible(actual, expected)
        assert_artifact_payloads(actual, output_dir)
        print(f"Loaded compatible forward artifact {actual.artifact_id}.")
        return _load_predictor(
            checkpoint_path, config, device, artifact_id=actual.artifact_id
        )

    training_seed = seed_registry.derive("forward/training", upper_bound=2**32)
    set_global_determinism(training_seed, strict=True, torch_num_threads=1)
    structural = bundle.structural_designs
    split_frames = {
        split: structural.loc[structural["split"] == split] for split in SPLIT_NAMES
    }
    if any(frame.empty for frame in split_frames.values()):
        raise ValueError("Forward training requires non-empty train/validation/test splits.")

    x_train = split_frames["train"].loc[:, DESIGN_COLUMNS].to_numpy(np.float64)
    y_train = split_frames["train"].loc[:, FORWARD_TARGET_COLUMNS].to_numpy(
        np.float64
    )
    input_mean = x_train.mean(axis=0)
    input_scale = x_train.std(axis=0, ddof=0)
    input_scale = np.where(input_scale > 1.0e-12, input_scale, 1.0)
    target_mean = y_train.mean(axis=0)
    if np.any(target_mean <= 0.0):
        raise ValueError("Positive-mean target scaling requires positive targets.")
    scalers = ForwardScalers(input_mean, input_scale, target_mean)

    def tensors(frame: pd.DataFrame) -> tuple[Any, Any]:
        x = frame.loc[:, DESIGN_COLUMNS].to_numpy(np.float64)
        y = frame.loc[:, FORWARD_TARGET_COLUMNS].to_numpy(np.float64)
        return (
            torch.as_tensor((x - input_mean) / input_scale, dtype=torch.float32),
            torch.as_tensor(y / target_mean, dtype=torch.float32),
        )

    x_train_tensor, y_train_tensor = tensors(split_frames["train"])
    x_validation_tensor, y_validation_tensor = tensors(split_frames["validation"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_registry.derive("forward/dataloader", upper_bound=2**32))
    loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )

    model = _make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.lr_patience,
        min_lr=config.min_learning_rate,
    )
    loss_function = nn.SmoothL1Loss(beta=config.huber_delta)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = loss_function(prediction, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.gradient_clip_norm
            )
            optimizer.step()
            train_total += float(loss.detach().cpu()) * len(x_batch)
            train_count += len(x_batch)

        model.eval()
        with torch.no_grad():
            validation_prediction = model(x_validation_tensor.to(device))
            validation_loss = float(
                loss_function(
                    validation_prediction, y_validation_tensor.to(device)
                ).cpu()
            )
        train_loss = train_total / train_count
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
        )
        if validation_loss < best_loss - 1.0e-12:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % config.print_every == 0:
            print(
                f"Forward epoch {epoch:4d}: train={train_loss:.6g}, "
                f"validation={validation_loss:.6g}, lr={learning_rate:.3g}"
            )
        if epochs_without_improvement >= config.patience:
            print(f"Forward early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Forward training did not produce a finite checkpoint.")
    model.load_state_dict(best_state)
    model.to(device).eval()

    metric_tables = []
    test_predictions = None
    for split, frame in split_frames.items():
        x_values = frame.loc[:, DESIGN_COLUMNS].to_numpy(np.float64)
        y_values = frame.loc[:, FORWARD_TARGET_COLUMNS].to_numpy(np.float64)
        predictions = _predict_array(model, x_values, scalers, device)
        table = regression_metrics(y_values, predictions, FORWARD_TARGET_COLUMNS)
        table.insert(0, "split", split)
        metric_tables.append(table)
        if split == "test":
            test_predictions = frame.loc[:, ["design_id", *DESIGN_COLUMNS]].copy()
            for target_index, target_name in enumerate(FORWARD_TARGET_COLUMNS):
                test_predictions[f"actual__{target_name}"] = y_values[:, target_index]
                test_predictions[f"predicted__{target_name}"] = predictions[
                    :, target_index
                ]
                test_predictions[f"residual__{target_name}"] = (
                    predictions[:, target_index] - y_values[:, target_index]
                )
    metric_table = pd.concat(metric_tables, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "scalers": scalers.to_dict(),
        "feature_names": DESIGN_COLUMNS,
        "target_names": FORWARD_TARGET_COLUMNS,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "artifact_id": expected.artifact_id,
    }
    with atomic_output_path(checkpoint_path) as temporary:
        torch.save(checkpoint, temporary)
    with atomic_output_path(history_path) as temporary:
        pd.DataFrame(history).to_csv(temporary, index=False)
    if test_predictions is None:
        raise RuntimeError("Forward test predictions were not materialized.")
    with atomic_output_path(test_predictions_path) as temporary:
        test_predictions.to_csv(temporary, index=False)
    atomic_write_json(
        metrics_path,
        {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "rows": json.loads(metric_table.to_json(orient="records")),
        },
    )
    persisted = expected.with_payload_hashes(
        output_dir,
        (
            checkpoint_path.name,
            metrics_path.name,
            history_path.name,
            test_predictions_path.name,
        ),
    )
    save_artifact_manifest(manifest_path, persisted)
    print(f"Trained forward artifact {expected.artifact_id}.")
    return ForwardPredictor(
        model, scalers, device=device, artifact_id=expected.artifact_id
    )
