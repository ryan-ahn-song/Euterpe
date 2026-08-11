from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
TASK_NAMES = ("pitch", "timbre", "melody", "naturalness")
TREATMENT_NAMES = (
    "reference",
    "harmonic_cue",
    "timbre_preserve",
    "melody_relief",
    "transient_preserve",
    "minimal_processing",
)
OUTCOME_NAMES = ("pitch", "timbre", "melody", "naturalness", "clarity")
MUSIC_CUE_NAMES = (
    "harmonicity",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_flux",
    "transient_ratio",
    "dynamic_range",
    "low_mid_energy_ratio",
    "high_energy_ratio",
    "f0_confidence",
    "normalized_f0",
)

_LISTENER_SET_FEATURES = 24
_RAW_RESPONSE_FEATURES = len(TASK_NAMES) + 7
_SET_RNG = np.random.default_rng(20_260_811)
_SET_PROJECTION = _SET_RNG.normal(
    0.0,
    1.0 / np.sqrt(_RAW_RESPONSE_FEATURES),
    size=(_RAW_RESPONSE_FEATURES, _LISTENER_SET_FEATURES),
).astype(np.float32)
_SET_BIAS = _SET_RNG.uniform(-0.5, 0.5, size=_LISTENER_SET_FEATURES).astype(np.float32)


def _finite_unit(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return number


@dataclass(frozen=True)
class PerceptionResponse:
    """One adaptive music-perception test item and its observed response."""

    task: str
    difficulty: float
    harmonicity: float
    polyphony: float
    masking: float
    correct: float
    confidence: float = 0.5
    reaction_time_ms: float = 1_500.0

    def validate(self) -> None:
        if self.task not in TASK_NAMES:
            raise ValueError(f"unknown task {self.task!r}; expected one of {TASK_NAMES}")
        for name in ("difficulty", "harmonicity", "polyphony", "masking", "correct", "confidence"):
            _finite_unit(getattr(self, name), name)
        if not np.isfinite(self.reaction_time_ms) or not 100.0 <= self.reaction_time_ms <= 20_000.0:
            raise ValueError("reaction_time_ms must be between 100 and 20000")

    def vector(self) -> FloatArray:
        self.validate()
        task = np.zeros(len(TASK_NAMES), dtype=np.float32)
        task[TASK_NAMES.index(self.task)] = 1.0
        response = np.asarray(
            [
                self.difficulty,
                self.harmonicity,
                self.polyphony,
                self.masking,
                self.correct,
                self.confidence,
                np.clip(np.log1p(self.reaction_time_ms) / np.log1p(20_000.0), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate((task, response)).astype(np.float32)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PerceptionResponse:
        item = cls(**raw)
        item.validate()
        return item


@dataclass(frozen=True)
class ListenerProfile:
    listener_id: str
    audiogram_db_hl: tuple[float, ...]
    responses: tuple[PerceptionResponse, ...]
    outcome_weights: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.listener_id.strip():
            raise ValueError("listener_id cannot be empty")
        audiogram = np.asarray(self.audiogram_db_hl, dtype=np.float32)
        if audiogram.ndim != 1 or audiogram.size < 2:
            raise ValueError("listener audiogram requires at least two values")
        if np.any(~np.isfinite(audiogram)) or np.any((audiogram < -10.0) | (audiogram > 120.0)):
            raise ValueError("listener audiogram values must lie between -10 and 120 dB HL")
        if not self.responses:
            raise ValueError("listener profile requires at least one perception response")
        for response in self.responses:
            response.validate()
        unknown = set(self.outcome_weights) - set(OUTCOME_NAMES)
        if unknown:
            raise ValueError(f"unknown outcome weight names: {sorted(unknown)}")
        for name, value in self.outcome_weights.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"outcome weight {name} must be finite and non-negative")

    def encoded(self) -> FloatArray:
        """DeepSets-style deterministic item encoder with mean and max pooling.

        Test items remain conditional on their stimulus properties rather than being
        reduced to one accuracy value per task. The downstream CAPE ensemble learns
        how these pooled nonlinear item features interact with music and treatment.
        """

        self.validate()
        audiogram = np.asarray(self.audiogram_db_hl, dtype=np.float32)
        source_axis = np.linspace(0.0, 1.0, audiogram.size)
        target_axis = np.linspace(0.0, 1.0, 8)
        audiogram_8 = np.interp(target_axis, source_axis, audiogram).astype(np.float32) / 120.0
        raw = np.stack([item.vector() for item in self.responses])
        hidden = np.maximum(raw @ _SET_PROJECTION + _SET_BIAS, 0.0)
        pooled = np.concatenate((np.mean(hidden, axis=0), np.max(hidden, axis=0)))
        return np.concatenate((audiogram_8, pooled)).astype(np.float32)

    def weights(self) -> FloatArray:
        if self.outcome_weights:
            values = np.asarray([self.outcome_weights.get(name, 0.0) for name in OUTCOME_NAMES], dtype=np.float32)
        else:
            values = []
            for task in OUTCOME_NAMES[:4]:
                items = [item for item in self.responses if item.task == task]
                if items:
                    performance = float(np.mean([item.correct for item in items]))
                    values.append(0.25 + (1.0 - performance))
                else:
                    values.append(0.5)
            values.append(0.35)  # clarity is not directly measured by the four core tasks
            values = np.asarray(values, dtype=np.float32)
        total = float(np.sum(values))
        if total <= 1e-8:
            return np.full(len(OUTCOME_NAMES), 1.0 / len(OUTCOME_NAMES), dtype=np.float32)
        return (values / total).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "listener_id": self.listener_id,
            "audiogram_db_hl": list(self.audiogram_db_hl),
            "responses": [asdict(item) for item in self.responses],
            "outcome_weights": dict(self.outcome_weights),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ListenerProfile:
        allowed = {"listener_id", "audiogram_db_hl", "responses", "outcome_weights"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown listener profile keys: {sorted(unknown)}")
        profile = cls(
            listener_id=str(raw["listener_id"]),
            audiogram_db_hl=tuple(float(value) for value in raw["audiogram_db_hl"]),
            responses=tuple(PerceptionResponse.from_dict(item) for item in raw["responses"]),
            outcome_weights={str(key): float(value) for key, value in raw.get("outcome_weights", {}).items()},
        )
        profile.validate()
        return profile

    @classmethod
    def load(cls, path: str | Path) -> ListenerProfile:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class MusicCueFeatures:
    harmonicity: float
    spectral_entropy: float
    spectral_flatness: float
    spectral_flux: float
    transient_ratio: float
    dynamic_range: float
    low_mid_energy_ratio: float
    high_energy_ratio: float
    f0_confidence: float
    normalized_f0: float

    def validate(self) -> None:
        for name in MUSIC_CUE_NAMES:
            _finite_unit(getattr(self, name), name)

    def vector(self) -> FloatArray:
        self.validate()
        return np.asarray([getattr(self, name) for name in MUSIC_CUE_NAMES], dtype=np.float32)

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in MUSIC_CUE_NAMES}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MusicCueFeatures:
        item = cls(**{name: float(raw[name]) for name in MUSIC_CUE_NAMES})
        item.validate()
        return item


@dataclass(frozen=True)
class Treatment:
    name: str
    strength: float = 0.0

    def validate(self) -> None:
        if self.name not in TREATMENT_NAMES:
            raise ValueError(f"unknown treatment {self.name!r}; expected one of {TREATMENT_NAMES}")
        _finite_unit(self.strength, "treatment strength")
        if self.name == "reference" and self.strength != 0.0:
            raise ValueError("reference treatment must have strength 0")

    def vector(self) -> FloatArray:
        self.validate()
        encoded = np.zeros(len(TREATMENT_NAMES) + 1, dtype=np.float32)
        encoded[TREATMENT_NAMES.index(self.name)] = 1.0
        encoded[-1] = self.strength
        return encoded

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "strength": self.strength}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Treatment:
        item = cls(name=str(raw["name"]), strength=float(raw.get("strength", 0.0)))
        item.validate()
        return item


@dataclass(frozen=True)
class OutcomeVector:
    pitch: float
    timbre: float
    melody: float
    naturalness: float
    clarity: float

    def validate(self) -> None:
        for name in OUTCOME_NAMES:
            _finite_unit(getattr(self, name), name)

    def vector(self) -> FloatArray:
        self.validate()
        return np.asarray([getattr(self, name) for name in OUTCOME_NAMES], dtype=np.float32)

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in OUTCOME_NAMES}

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> OutcomeVector:
        item = cls(**dict(zip(OUTCOME_NAMES, (float(value) for value in values), strict=True)))
        item.validate()
        return item

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> OutcomeVector:
        return cls.from_vector([raw[name] for name in OUTCOME_NAMES])


@dataclass(frozen=True)
class CAPEExample:
    listener: ListenerProfile
    music: MusicCueFeatures
    treatment: Treatment
    outcome: OutcomeVector

    def validate(self) -> None:
        self.listener.validate()
        self.music.validate()
        self.treatment.validate()
        self.outcome.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "listener": self.listener.to_dict(),
            "music": self.music.to_dict(),
            "treatment": self.treatment.to_dict(),
            "outcome": self.outcome.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CAPEExample:
        item = cls(
            listener=ListenerProfile.from_dict(raw["listener"]),
            music=MusicCueFeatures.from_dict(raw["music"]),
            treatment=Treatment.from_dict(raw["treatment"]),
            outcome=OutcomeVector.from_dict(raw["outcome"]),
        )
        item.validate()
        return item


def load_examples(path: str | Path) -> list[CAPEExample]:
    examples: list[CAPEExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(CAPEExample.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid CAPE JSONL at line {line_number}: {exc}") from exc
    if not examples:
        raise ValueError("CAPE dataset is empty")
    return examples


def save_examples(path: str | Path, examples: Iterable[CAPEExample]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            example.validate()
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("refusing to write an empty CAPE dataset")
    return count


@dataclass(frozen=True)
class EffectPrediction:
    treatment: Treatment
    outcome: OutcomeVector
    reference: OutcomeVector
    delta: tuple[float, ...]
    uncertainty: float


class CAPEEffectModel:
    """Small bootstrap random-feature ensemble for counterfactual outcomes.

    This model is intentionally compact enough for real-time CPU inference. Each
    ensemble member is a nonlinear random-feature regressor fitted to a bootstrap
    sample. Dispersion between members is exposed as uncertainty and used by the
    policy to abstain. It is a research model, not a clinical predictor.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        ensemble_size: int = 7,
        hidden_features: int = 96,
        ridge: float = 0.1,
        seed: int = 20_260_811,
    ) -> None:
        if not 3 <= ensemble_size <= 31:
            raise ValueError("ensemble_size must be between 3 and 31")
        if not 8 <= hidden_features <= 1_024:
            raise ValueError("hidden_features must be between 8 and 1024")
        if not 1e-6 <= ridge <= 1e3:
            raise ValueError("ridge must be between 1e-6 and 1e3")
        self.ensemble_size = int(ensemble_size)
        self.hidden_features = int(hidden_features)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self._mean: FloatArray | None = None
        self._scale: FloatArray | None = None
        self._projection: FloatArray | None = None
        self._bias: FloatArray | None = None
        self._coefficients: FloatArray | None = None
        self._training_radius: float = 1.0
        self.training_metadata: dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        return self._coefficients is not None

    @staticmethod
    def feature_vector(listener: ListenerProfile, music: MusicCueFeatures, treatment: Treatment) -> FloatArray:
        return np.concatenate((listener.encoded(), music.vector(), treatment.vector())).astype(np.float32)

    def _matrix(self, examples: Sequence[CAPEExample]) -> tuple[FloatArray, FloatArray]:
        if not examples:
            raise ValueError("at least one CAPE example is required")
        for example in examples:
            example.validate()
        x = np.stack(
            [self.feature_vector(example.listener, example.music, example.treatment) for example in examples]
        ).astype(np.float32)
        y = np.stack([example.outcome.vector() for example in examples]).astype(np.float32)
        return x, y

    def fit(self, examples: Sequence[CAPEExample]) -> dict[str, float]:
        if len(examples) < 12:
            raise ValueError("CAPE training requires at least 12 observations")
        observed_treatments = {example.treatment.name for example in examples}
        if "reference" not in observed_treatments:
            raise ValueError("CAPE training data requires reference observations")
        if len(observed_treatments) < 2:
            raise ValueError("CAPE training data requires at least one non-reference treatment")
        x, y = self._matrix(examples)
        self._mean = np.mean(x, axis=0).astype(np.float32)
        scale = np.std(x, axis=0)
        self._scale = np.where(scale < 1e-4, 1.0, scale).astype(np.float32)
        normalized = (x - self._mean) / self._scale
        self._training_radius = float(np.percentile(np.linalg.norm(normalized, axis=1), 95))

        rng = np.random.default_rng(self.seed)
        input_size = x.shape[1]
        projection = rng.normal(
            0.0,
            1.0 / np.sqrt(input_size),
            size=(self.ensemble_size, input_size, self.hidden_features),
        ).astype(np.float32)
        bias = rng.uniform(-1.0, 1.0, size=(self.ensemble_size, self.hidden_features)).astype(np.float32)
        coefficient_count = 1 + input_size + self.hidden_features
        coefficients = np.empty(
            (self.ensemble_size, coefficient_count, len(OUTCOME_NAMES)), dtype=np.float32
        )
        predictions = np.empty((self.ensemble_size, x.shape[0], len(OUTCOME_NAMES)), dtype=np.float32)
        for member in range(self.ensemble_size):
            indices = rng.integers(0, x.shape[0], size=x.shape[0])
            boot_x = normalized[indices]
            boot_y = y[indices]
            hidden = np.maximum(boot_x @ projection[member] + bias[member], 0.0)
            design = np.concatenate(
                (
                    np.ones((boot_x.shape[0], 1), dtype=np.float32),
                    boot_x,
                    hidden / np.sqrt(self.hidden_features),
                ),
                axis=1,
            )
            penalty = self.ridge * np.eye(coefficient_count, dtype=np.float32)
            penalty[0, 0] = 0.0
            lhs = design.T @ design + penalty
            rhs = design.T @ boot_y
            coefficients[member] = np.linalg.solve(lhs, rhs).astype(np.float32)
            full_hidden = np.maximum(normalized @ projection[member] + bias[member], 0.0)
            full_design = np.concatenate(
                (
                    np.ones((normalized.shape[0], 1), dtype=np.float32),
                    normalized,
                    full_hidden / np.sqrt(self.hidden_features),
                ),
                axis=1,
            )
            predictions[member] = full_design @ coefficients[member]

        self._projection = projection
        self._bias = bias
        self._coefficients = coefficients
        mean_prediction = np.clip(np.mean(predictions, axis=0), 0.0, 1.0)
        metrics = _regression_metrics(y, mean_prediction)
        self.training_metadata = {
            "observations": len(examples),
            "listeners": len({example.listener.listener_id for example in examples}),
            "treatments": sorted(observed_treatments),
            "strength_ranges": {
                name: [
                    float(min(example.treatment.strength for example in examples if example.treatment.name == name)),
                    float(max(example.treatment.strength for example in examples if example.treatment.name == name)),
                ]
                for name in sorted(observed_treatments)
            },
            "training_metrics": metrics,
        }
        return metrics

    def _member_prediction_batch(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        if any(value is None for value in (self._mean, self._scale, self._projection, self._bias, self._coefficients)):
            raise RuntimeError("CAPE model is not fitted")
        matrix = np.atleast_2d(features).astype(np.float32)
        normalized = (matrix - self._mean) / self._scale  # type: ignore[operator]
        hidden = np.maximum(
            np.einsum("nd,edh->enh", normalized, self._projection)  # type: ignore[arg-type]
            + self._bias[:, None, :],  # type: ignore[index]
            0.0,
        )
        repeated = np.broadcast_to(
            normalized[None, :, :], (self.ensemble_size, normalized.shape[0], normalized.shape[1])
        )
        design = np.concatenate(
            (
                np.ones((self.ensemble_size, normalized.shape[0], 1), dtype=np.float32),
                repeated,
                hidden / np.sqrt(self.hidden_features),
            ),
            axis=2,
        )
        predictions = np.einsum("enc,eco->eno", design, self._coefficients)  # type: ignore[arg-type]
        return np.clip(predictions, 0.0, 1.0).astype(np.float32)

    def _member_predictions(self, feature: FloatArray) -> NDArray[np.float32]:
        return self._member_prediction_batch(feature)[:, 0, :]

    def predict_effect(
        self,
        listener: ListenerProfile,
        music: MusicCueFeatures,
        treatment: Treatment,
    ) -> EffectPrediction:
        treatment.validate()
        reference = Treatment("reference", 0.0)
        candidate_members = self._member_predictions(self.feature_vector(listener, music, treatment))
        reference_members = self._member_predictions(self.feature_vector(listener, music, reference))
        candidate_mean = np.mean(candidate_members, axis=0)
        reference_mean = np.mean(reference_members, axis=0)
        delta_members = candidate_members - reference_members
        delta_mean = candidate_mean - reference_mean

        feature = self.feature_vector(listener, music, treatment)
        normalized = (feature - self._mean) / self._scale  # type: ignore[operator]
        radius = float(np.linalg.norm(normalized))
        extrapolation = max(0.0, radius - self._training_radius) / max(self._training_radius, 1e-6)
        uncertainty = float(np.mean(np.std(delta_members, axis=0)) + 0.1 * min(1.0, extrapolation))
        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
        return EffectPrediction(
            treatment=treatment,
            outcome=OutcomeVector.from_vector(np.clip(candidate_mean, 0.0, 1.0)),
            reference=OutcomeVector.from_vector(np.clip(reference_mean, 0.0, 1.0)),
            delta=tuple(float(value) for value in np.clip(delta_mean, -1.0, 1.0)),
            uncertainty=uncertainty,
        )

    def raw_effect(
        self,
        listener: ListenerProfile,
        music: MusicCueFeatures,
        treatment: Treatment,
    ) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        """Return candidate, reference, signed delta, and uncertainty arrays."""

        return self.raw_effects(listener, music, [treatment])[0]

    def raw_effects(
        self,
        listener: ListenerProfile,
        music: MusicCueFeatures,
        treatments: Sequence[Treatment],
    ) -> list[tuple[FloatArray, FloatArray, FloatArray, float]]:
        """Vectorized counterfactual predictions for all candidates in one inference."""

        if not treatments:
            return []
        reference = Treatment("reference", 0.0)
        all_treatments = [reference, *treatments]
        shared = np.concatenate((listener.encoded(), music.vector())).astype(np.float32)
        treatment_matrix = np.stack([treatment.vector() for treatment in all_treatments])
        features = np.concatenate(
            (np.broadcast_to(shared, (len(all_treatments), shared.size)), treatment_matrix), axis=1
        ).astype(np.float32)
        member_predictions = self._member_prediction_batch(features)
        reference_members = member_predictions[:, 0, :]
        baseline = np.mean(reference_members, axis=0).astype(np.float32)
        normalized = (features - self._mean) / self._scale  # type: ignore[operator]
        results: list[tuple[FloatArray, FloatArray, FloatArray, float]] = []
        for index in range(1, len(all_treatments)):
            candidate_members = member_predictions[:, index, :]
            candidate = np.mean(candidate_members, axis=0).astype(np.float32)
            delta_members = candidate_members - reference_members
            radius = float(np.linalg.norm(normalized[index]))
            extrapolation = max(0.0, radius - self._training_radius) / max(self._training_radius, 1e-6)
            uncertainty = float(
                np.clip(
                    np.mean(np.std(delta_members, axis=0)) + 0.1 * min(1.0, extrapolation),
                    0.0,
                    1.0,
                )
            )
            results.append(
                (candidate, baseline.copy(), (candidate - baseline).astype(np.float32), uncertainty)
            )
        return results

    def evaluate(self, examples: Sequence[CAPEExample]) -> dict[str, float]:
        x, expected = self._matrix(examples)
        predicted = np.mean(self._member_prediction_batch(x), axis=0)
        return _regression_metrics(expected, predicted)

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("cannot save an unfitted CAPE model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": self.FORMAT_VERSION,
            "ensemble_size": self.ensemble_size,
            "hidden_features": self.hidden_features,
            "ridge": self.ridge,
            "seed": self.seed,
            "outcome_names": OUTCOME_NAMES,
            "treatment_names": TREATMENT_NAMES,
            "music_cue_names": MUSIC_CUE_NAMES,
            "training": self.training_metadata,
        }
        with destination.open("wb") as handle:
            np.savez_compressed(
                handle,
                mean=self._mean,
                scale=self._scale,
                projection=self._projection,
                bias=self._bias,
                coefficients=self._coefficients,
                training_radius=np.asarray([self._training_radius], dtype=np.float32),
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )

    @classmethod
    def load(cls, path: str | Path) -> CAPEEffectModel:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError("unsupported CAPE model format")
            if tuple(metadata.get("outcome_names", ())) != OUTCOME_NAMES:
                raise ValueError("CAPE model outcome schema mismatch")
            if tuple(metadata.get("treatment_names", ())) != TREATMENT_NAMES:
                raise ValueError("CAPE model treatment schema mismatch")
            model = cls(
                ensemble_size=int(metadata["ensemble_size"]),
                hidden_features=int(metadata["hidden_features"]),
                ridge=float(metadata["ridge"]),
                seed=int(metadata["seed"]),
            )
            model._mean = archive["mean"].astype(np.float32)
            model._scale = archive["scale"].astype(np.float32)
            model._projection = archive["projection"].astype(np.float32)
            model._bias = archive["bias"].astype(np.float32)
            model._coefficients = archive["coefficients"].astype(np.float32)
            model._training_radius = float(archive["training_radius"][0])
            model.training_metadata = dict(metadata.get("training", {}))
        expected_input = 8 + 2 * _LISTENER_SET_FEATURES + len(MUSIC_CUE_NAMES) + len(TREATMENT_NAMES) + 1
        if model._mean.size != expected_input:
            raise ValueError("CAPE model input schema mismatch")
        return model


def _regression_metrics(expected: NDArray[np.float32], predicted: NDArray[np.float32]) -> dict[str, float]:
    error = predicted - expected
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "brier": float(np.mean(error**2)),
    }


@dataclass(frozen=True)
class PolicySettings:
    strengths: tuple[float, ...] = (0.35, 0.65, 1.0)
    min_effect: float = 0.015
    max_uncertainty: float = 0.20
    max_distortion: float = 0.18
    max_naturalness_drop: float = 0.08
    distortion_penalty: float = 0.35
    uncertainty_penalty: float = 0.25


@dataclass(frozen=True)
class PolicyDecision:
    treatment: Treatment
    score: float
    uncertainty: float
    estimated_distortion: float
    delta: tuple[float, ...]
    reason: str

    @classmethod
    def reference(cls, reason: str) -> PolicyDecision:
        return cls(
            treatment=Treatment("reference", 0.0),
            score=0.0,
            uncertainty=0.0,
            estimated_distortion=0.0,
            delta=(0.0,) * len(OUTCOME_NAMES),
            reason=reason,
        )


class CAPEPolicy:
    def __init__(self, model: CAPEEffectModel, listener: ListenerProfile, settings: PolicySettings) -> None:
        if not model.fitted:
            raise ValueError("CAPE policy requires a fitted model")
        listener.validate()
        self.model = model
        self.listener = listener
        self.settings = settings
        self.weights = listener.weights()

    @staticmethod
    def estimated_distortion(treatment: Treatment, music: MusicCueFeatures) -> float:
        strength = treatment.strength
        if treatment.name == "reference":
            return 0.0
        if treatment.name == "harmonic_cue":
            return strength * (0.025 + 0.10 * (1.0 - music.harmonicity))
        if treatment.name == "timbre_preserve":
            return strength * 0.035
        if treatment.name == "melody_relief":
            return strength * (0.05 + 0.10 * music.spectral_entropy)
        if treatment.name == "transient_preserve":
            return strength * (0.035 + 0.07 * (1.0 - music.transient_ratio))
        return strength * 0.01

    def select(self, music: MusicCueFeatures) -> PolicyDecision:
        music.validate()
        accepted: list[PolicyDecision] = []
        supported = set(self.model.training_metadata.get("treatments", ()))
        ranges = self.model.training_metadata.get("strength_ranges", {})
        candidates: list[Treatment] = []
        for name in (item for item in TREATMENT_NAMES[1:] if item in supported):
            for strength in self.settings.strengths:
                if name in ranges and not float(ranges[name][0]) <= strength <= float(ranges[name][1]):
                    continue
                candidates.append(Treatment(name, strength))
        effects = self.model.raw_effects(self.listener, music, candidates)
        for treatment, (_, _, delta, uncertainty) in zip(candidates, effects, strict=True):
            distortion = self.estimated_distortion(treatment, music)
            score = float(
                np.dot(self.weights, delta)
                - self.settings.distortion_penalty * distortion
                - self.settings.uncertainty_penalty * uncertainty
            )
            if (
                uncertainty <= self.settings.max_uncertainty
                and distortion <= self.settings.max_distortion
                and delta[OUTCOME_NAMES.index("naturalness")]
                >= -self.settings.max_naturalness_drop
            ):
                accepted.append(
                    PolicyDecision(
                        treatment=treatment,
                        score=score,
                        uncertainty=uncertainty,
                        estimated_distortion=distortion,
                        delta=tuple(float(value) for value in delta),
                        reason="predicted_positive_net_effect",
                    )
                )
        if not accepted:
            return PolicyDecision.reference("all_candidates_rejected_by_constraints")
        best = max(accepted, key=lambda decision: decision.score)
        if best.score < self.settings.min_effect:
            return PolicyDecision.reference("predicted_effect_below_threshold")
        return best
