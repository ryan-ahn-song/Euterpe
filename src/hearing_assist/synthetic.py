from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cape import (
    CAPEExample,
    ListenerProfile,
    MusicCueFeatures,
    OutcomeVector,
    PerceptionResponse,
    TASK_NAMES,
    TREATMENT_NAMES,
    Treatment,
)


@dataclass(frozen=True)
class SyntheticListener:
    profile: ListenerProfile
    pitch_deficit: float
    timbre_deficit: float
    melody_deficit: float
    naturalness_sensitivity: float


def _listener(index: int, rng: np.random.Generator) -> SyntheticListener:
    deficits = rng.uniform(0.15, 0.85, size=3)
    naturalness_sensitivity = float(rng.uniform(0.2, 0.9))
    slope = float(rng.uniform(10.0, 45.0))
    base = float(rng.uniform(10.0, 35.0))
    audiogram = np.clip(base + slope * np.linspace(0.0, 1.0, 8) + rng.normal(0.0, 2.0, 8), 0.0, 85.0)
    responses: list[PerceptionResponse] = []
    task_deficits = {
        "pitch": deficits[0],
        "timbre": deficits[1],
        "melody": deficits[2],
        "naturalness": naturalness_sensitivity * 0.5,
    }
    for task in TASK_NAMES:
        for _ in range(8):
            difficulty = float(rng.uniform(0.1, 1.0))
            harmonicity = float(rng.uniform(0.0, 1.0))
            polyphony = float(rng.uniform(0.0, 1.0))
            masking = float(rng.uniform(0.0, 1.0))
            context_penalty = 0.20 * difficulty + 0.12 * masking
            if task == "pitch":
                context_penalty += 0.15 * (1.0 - harmonicity)
            elif task == "melody":
                context_penalty += 0.18 * polyphony
            probability = np.clip(1.0 - 0.75 * task_deficits[task] - context_penalty, 0.05, 0.98)
            correct = float(rng.random() < probability)
            confidence = float(np.clip(probability + rng.normal(0.0, 0.12), 0.0, 1.0))
            reaction = float(np.clip(700.0 + 2_000.0 * difficulty + 1_200.0 * masking + rng.normal(0, 160), 250, 8_000))
            responses.append(
                PerceptionResponse(
                    task=task,
                    difficulty=difficulty,
                    harmonicity=harmonicity,
                    polyphony=polyphony,
                    masking=masking,
                    correct=correct,
                    confidence=confidence,
                    reaction_time_ms=reaction,
                )
            )
    profile = ListenerProfile(
        listener_id=f"synthetic-listener-{index:02d}",
        audiogram_db_hl=tuple(float(value) for value in audiogram),
        responses=tuple(responses),
    )
    return SyntheticListener(
        profile=profile,
        pitch_deficit=float(deficits[0]),
        timbre_deficit=float(deficits[1]),
        melody_deficit=float(deficits[2]),
        naturalness_sensitivity=naturalness_sensitivity,
    )


def _music(rng: np.random.Generator) -> MusicCueFeatures:
    harmonicity = float(rng.beta(2.0, 1.8))
    entropy = float(rng.beta(2.0, 2.0))
    flatness = float(np.clip(0.65 * entropy + rng.normal(0.0, 0.12), 0.0, 1.0))
    flux = float(rng.beta(1.6, 3.0))
    transient = float(np.clip(0.75 * flux + rng.normal(0.0, 0.15), 0.0, 1.0))
    return MusicCueFeatures(
        harmonicity=harmonicity,
        spectral_entropy=entropy,
        spectral_flatness=flatness,
        spectral_flux=flux,
        transient_ratio=transient,
        dynamic_range=float(rng.beta(2.0, 2.0)),
        low_mid_energy_ratio=float(rng.beta(2.5, 2.0)),
        high_energy_ratio=float(rng.beta(1.5, 3.0)),
        f0_confidence=float(np.clip(harmonicity + rng.normal(0.0, 0.08), 0.0, 1.0)),
        normalized_f0=float(rng.uniform(0.0, 1.0)),
    )


def _outcome(
    listener: SyntheticListener,
    music: MusicCueFeatures,
    treatment: Treatment,
    rng: np.random.Generator,
) -> OutcomeVector:
    pitch = 0.78 - listener.pitch_deficit * (0.20 + 0.30 * (1.0 - music.harmonicity))
    timbre = 0.76 - listener.timbre_deficit * (0.18 + 0.28 * music.spectral_entropy)
    melody = 0.74 - listener.melody_deficit * (0.20 + 0.34 * music.spectral_entropy)
    naturalness = 0.84 - 0.08 * music.spectral_flatness
    clarity = 0.73 - 0.20 * music.spectral_entropy
    strength = treatment.strength

    if treatment.name == "harmonic_cue":
        pitch += 0.34 * listener.pitch_deficit * music.harmonicity * strength
        clarity += 0.06 * music.harmonicity * strength
        naturalness -= 0.15 * listener.naturalness_sensitivity * (1.0 - music.harmonicity) * strength
    elif treatment.name == "timbre_preserve":
        timbre += 0.27 * listener.timbre_deficit * (0.4 + 0.6 * music.harmonicity) * strength
        naturalness += 0.04 * listener.naturalness_sensitivity * strength
    elif treatment.name == "melody_relief":
        melody += 0.36 * listener.melody_deficit * music.low_mid_energy_ratio * strength
        clarity += 0.16 * (1.0 - music.spectral_flatness) * strength
        naturalness -= 0.12 * listener.naturalness_sensitivity * music.spectral_entropy * strength
    elif treatment.name == "transient_preserve":
        timbre += 0.12 * listener.timbre_deficit * music.transient_ratio * strength
        clarity += 0.10 * music.transient_ratio * strength
        naturalness += 0.05 * listener.naturalness_sensitivity * music.transient_ratio * strength
    elif treatment.name == "minimal_processing":
        naturalness += 0.09 * listener.naturalness_sensitivity * strength
        clarity -= 0.04 * listener.melody_deficit * strength

    values = np.asarray([pitch, timbre, melody, naturalness, clarity], dtype=np.float32)
    values += rng.normal(0.0, 0.018, size=values.size).astype(np.float32)
    return OutcomeVector.from_vector(np.clip(values, 0.0, 1.0))


def generate_synthetic_examples(
    listener_count: int = 8,
    music_segments_per_listener: int = 8,
    seed: int = 20_260_811,
) -> list[CAPEExample]:
    """Generate engineering-only observations for pipeline smoke testing.

    These outcomes are equations authored for software verification. They must not
    be presented as simulated physiology, clinical data, or evidence of benefit.
    """

    if listener_count < 1 or music_segments_per_listener < 1:
        raise ValueError("listener and segment counts must be positive")
    rng = np.random.default_rng(seed)
    strengths = (0.35, 0.65, 1.0)
    examples: list[CAPEExample] = []
    for listener_index in range(listener_count):
        listener = _listener(listener_index, rng)
        for _ in range(music_segments_per_listener):
            music = _music(rng)
            treatments = [Treatment("reference", 0.0)]
            treatments.extend(
                Treatment(name, strength)
                for name in TREATMENT_NAMES[1:]
                for strength in strengths
            )
            for treatment in treatments:
                examples.append(
                    CAPEExample(
                        listener=listener.profile,
                        music=music,
                        treatment=treatment,
                        outcome=_outcome(listener, music, treatment, rng),
                    )
                )
    return examples
