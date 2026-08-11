from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import butter, sosfilt

from music_ci.analysis.pitch import CausalPitchTracker

from .cape import (
    CAPEEffectModel,
    CAPEPolicy,
    ListenerProfile,
    MusicCueFeatures,
    PolicyDecision,
    PolicySettings,
    Treatment,
)
from .config import HearingAssistConfig

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class BlockDiagnostics:
    runtime_ms: float
    input_peak_dbfs: float
    output_peak_dbfs: float
    band_levels_dbfs: FloatArray
    applied_gain_db: FloatArray
    f0_hz: float
    f0_confidence: float
    cape_action: str
    cape_strength: float
    cape_score: float
    cape_uncertainty: float
    cape_estimated_distortion: float
    cape_delta_pitch: float
    cape_delta_timbre: float
    cape_delta_melody: float
    cape_delta_naturalness: float
    cape_delta_clarity: float
    cape_reason: str
    limiter_reduction_db: float
    fail_safe_triggered: bool


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


class HearingAssistProcessor:
    """Streaming WOLA processor with WDRC, music cues, and a digital limiter.

    Output is delayed by one block. The limiter ceiling is digital dBFS only;
    acoustic SPL still depends on the DAC, amplifier, receiver, and ear coupling.
    """

    def __init__(self, config: HearingAssistConfig) -> None:
        config.validate()
        self.config = config
        self.sample_rate = config.audio.sample_rate
        self.hop = config.audio.block_size
        self.fft_size = 2 * self.hop
        self._window = np.sqrt(np.hanning(self.fft_size + 1)[:-1]).astype(np.float64)
        self._window_sum = float(np.sum(self._window))
        self._frequencies = rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        self._profile_frequencies = np.asarray(config.audiogram.frequencies_hz, dtype=np.float64)
        self._base_gain_db = config.prescribed_gain_db().astype(np.float64)
        self._bin_bands = self._build_band_membership()

        self._previous_input = np.zeros(self.hop, dtype=np.float64)
        self._overlap = np.zeros(self.hop, dtype=np.float64)
        self._smoothed_gain_db = np.zeros_like(self._base_gain_db)
        self._limiter_gain = 1.0
        self._processed_samples = 0
        self._fail_safe_count = 0
        self._previous_magnitude = np.zeros(self._frequencies.size, dtype=np.float64)

        self._cape_policy: CAPEPolicy | None = None
        self._cape_decision = PolicyDecision.reference("cape_disabled")
        if config.cape.enabled:
            model = CAPEEffectModel.load(config.cape.model_path)
            listener = ListenerProfile.load(config.cape.listener_profile_path)
            configured_audiogram = np.asarray(config.audiogram.hearing_loss_db_hl, dtype=np.float32)
            listener_audiogram = np.asarray(listener.audiogram_db_hl, dtype=np.float32)
            if configured_audiogram.shape != listener_audiogram.shape or not np.allclose(
                configured_audiogram, listener_audiogram, atol=1e-3
            ):
                raise ValueError(
                    "CAPE listener audiogram must exactly match the active hearing profile"
                )
            settings = PolicySettings(
                strengths=tuple(config.cape.strengths),
                min_effect=config.cape.min_effect,
                max_uncertainty=config.cape.max_uncertainty,
                max_distortion=config.cape.max_distortion,
                max_naturalness_drop=config.cape.max_naturalness_drop,
                distortion_penalty=config.cape.distortion_penalty,
                uncertainty_penalty=config.cape.uncertainty_penalty,
            )
            self._cape_policy = CAPEPolicy(model, listener, settings)
        self._cape_active = Treatment("reference", 0.0)
        self._cape_previous = Treatment("reference", 0.0)
        self._cape_decision_counter = config.cape.decision_interval_blocks
        self._cape_hold_blocks = int(
            np.ceil(config.cape.min_hold_ms * self.sample_rate / (1_000.0 * self.hop))
        )
        self._cape_blocks_since_switch = self._cape_hold_blocks
        self._cape_transition_blocks = max(
            1,
            int(np.ceil(config.cape.crossfade_ms * self.sample_rate / (1_000.0 * self.hop))),
        )
        self._cape_transition_position = self._cape_transition_blocks

        decimation = max(1, int(round(self.sample_rate / 12_000)))
        self._pitch_decimation = decimation
        pitch_rate = self.sample_rate // decimation
        cutoff = min(3_000.0, 0.42 * pitch_rate)
        self._pitch_sos = butter(2, cutoff, btype="lowpass", fs=self.sample_rate, output="sos")
        self._pitch_zi = np.zeros((self._pitch_sos.shape[0], 2), dtype=np.float64)
        self._pitch_tracker = CausalPitchTracker(
            sample_rate=pitch_rate,
            context_ms=48,
            min_hz=config.music.pitch_min_hz,
            max_hz=config.music.pitch_max_hz,
        )

    @property
    def fail_safe_count(self) -> int:
        return self._fail_safe_count

    @property
    def algorithmic_latency_ms(self) -> float:
        return 1_000.0 * self.hop / self.sample_rate

    def _build_band_membership(self) -> list[np.ndarray]:
        centers = self._profile_frequencies
        boundaries = np.sqrt(centers[:-1] * centers[1:])
        lower = np.concatenate(([0.0], boundaries))
        upper = np.concatenate((boundaries, [self.sample_rate / 2 + 1.0]))
        return [np.flatnonzero((self._frequencies >= lo) & (self._frequencies < hi)) for lo, hi in zip(lower, upper)]

    def reset(self) -> None:
        self._previous_input.fill(0.0)
        self._overlap.fill(0.0)
        self._smoothed_gain_db.fill(0.0)
        self._limiter_gain = 1.0
        self._processed_samples = 0
        self._fail_safe_count = 0
        self._previous_magnitude.fill(0.0)
        self._pitch_zi.fill(0.0)
        self._pitch_tracker.reset()
        self._cape_decision = PolicyDecision.reference(
            "waiting_for_first_cape_decision" if self._cape_policy is not None else "cape_disabled"
        )
        self._cape_active = Treatment("reference", 0.0)
        self._cape_previous = Treatment("reference", 0.0)
        self._cape_decision_counter = self.config.cape.decision_interval_blocks
        self._cape_blocks_since_switch = self._cape_hold_blocks
        self._cape_transition_position = self._cape_transition_blocks

    def _band_levels(self, spectrum: NDArray[np.complex128]) -> NDArray[np.float64]:
        normalized = 2.0 * np.abs(spectrum) / max(self._window_sum, 1e-12)
        levels = np.empty(len(self._bin_bands), dtype=np.float64)
        for index, bins in enumerate(self._bin_bands):
            if bins.size == 0:
                levels[index] = -120.0
            else:
                rms = float(np.sqrt(0.5 * np.sum(normalized[bins] ** 2)))
                levels[index] = max(-120.0, _db(rms))
        return levels

    def _target_gains(self, levels_dbfs: NDArray[np.float64]) -> NDArray[np.float64]:
        c = self.config.compression
        over_knee = np.maximum(levels_dbfs - c.knee_dbfs, 0.0)
        compression_reduction = (1.0 - 1.0 / c.ratio) * over_knee
        target = np.clip(self._base_gain_db - compression_reduction, 0.0, c.max_gain_db)
        target = np.where(levels_dbfs < c.low_level_gate_dbfs, 0.0, target)
        return target

    def _smooth_gains(
        self, target_db: NDArray[np.float64], attack_scale: float = 1.0
    ) -> NDArray[np.float64]:
        c = self.config.compression
        block_seconds = self.hop / self.sample_rate
        attack = np.exp(-block_seconds / ((c.attack_ms * attack_scale) / 1_000.0))
        release = np.exp(-block_seconds / (c.release_ms / 1_000.0))
        coefficient = np.where(target_db < self._smoothed_gain_db, attack, release)
        self._smoothed_gain_db = coefficient * self._smoothed_gain_db + (1.0 - coefficient) * target_db
        return self._smoothed_gain_db

    def _pitch(self, block: NDArray[np.float64]) -> tuple[float, float]:
        if not self.config.music.enabled:
            return 0.0, 0.0
        filtered, self._pitch_zi = sosfilt(self._pitch_sos, block, zi=self._pitch_zi)
        decimated = filtered[:: self._pitch_decimation].astype(np.float32)
        f0_hz, confidence, _ = self._pitch_tracker.update(decimated)
        return f0_hz, confidence

    def _base_gain_curve_db(self, band_gain_db: NDArray[np.float64]) -> NDArray[np.float64]:
        positive_frequency = np.maximum(self._frequencies, self._profile_frequencies[0])
        interpolation_frequencies = np.concatenate(
            (self._profile_frequencies, [self.sample_rate / 2.0])
        )
        interpolation_gains = np.concatenate((band_gain_db, [0.0]))
        return np.interp(
            np.log2(positive_frequency),
            np.log2(interpolation_frequencies),
            interpolation_gains,
            left=band_gain_db[0],
            right=0.0,
        )

    def _music_cues(
        self,
        spectrum: NDArray[np.complex128],
        block: NDArray[np.float64],
        f0_hz: float,
        confidence: float,
    ) -> MusicCueFeatures:
        magnitude = np.abs(spectrum).astype(np.float64)
        positive = magnitude[1:] + 1e-12
        probability = positive / max(float(np.sum(positive)), 1e-12)
        entropy = -float(np.sum(probability * np.log(probability))) / max(np.log(probability.size), 1.0)
        flatness = float(np.exp(np.mean(np.log(positive))) / max(np.mean(positive), 1e-12))
        flux = float(
            np.sum(np.maximum(magnitude - self._previous_magnitude, 0.0))
            / max(np.sum(magnitude), 1e-12)
        )
        self._previous_magnitude = magnitude
        rms = float(np.sqrt(np.mean(block**2)))
        peak = float(np.max(np.abs(block))) if block.size else 0.0
        crest = peak / max(rms, 1e-12)
        power = magnitude**2
        total = max(float(np.sum(power)), 1e-12)
        low_mid = float(np.sum(power[(self._frequencies >= 150.0) & (self._frequencies < 2_000.0)])) / total
        high = float(np.sum(power[self._frequencies >= 4_000.0])) / total
        if f0_hz > 0.0:
            log_range = np.log2(self.config.music.pitch_max_hz / self.config.music.pitch_min_hz)
            normalized_f0 = np.log2(f0_hz / self.config.music.pitch_min_hz) / max(log_range, 1e-6)
        else:
            normalized_f0 = 0.0
        return MusicCueFeatures(
            harmonicity=float(np.clip(confidence, 0.0, 1.0)),
            spectral_entropy=float(np.clip(entropy, 0.0, 1.0)),
            spectral_flatness=float(np.clip(flatness, 0.0, 1.0)),
            spectral_flux=float(np.clip(2.0 * flux, 0.0, 1.0)),
            transient_ratio=float(np.clip((crest - 1.0) / 5.0, 0.0, 1.0)),
            dynamic_range=float(np.clip((crest - 1.0) / 7.0, 0.0, 1.0)),
            low_mid_energy_ratio=float(np.clip(low_mid, 0.0, 1.0)),
            high_energy_ratio=float(np.clip(high, 0.0, 1.0)),
            f0_confidence=float(np.clip(confidence, 0.0, 1.0)),
            normalized_f0=float(np.clip(normalized_f0, 0.0, 1.0)),
        )

    def _select_treatment(
        self, cues: MusicCueFeatures
    ) -> tuple[PolicyDecision, Treatment, Treatment, float]:
        if self._cape_policy is None:
            if self.config.music.enabled and self.config.music.harmonic_boost_db > 0.0:
                legacy = Treatment("harmonic_cue", 1.0)
                decision = PolicyDecision(
                    treatment=legacy,
                    score=0.0,
                    uncertainty=0.0,
                    estimated_distortion=0.0,
                    delta=(0.0,) * 5,
                    reason="legacy_confidence_gated_harmonic_baseline",
                )
                return decision, legacy, legacy, 1.0
            decision = PolicyDecision.reference("cape_and_music_processing_disabled")
            return decision, decision.treatment, decision.treatment, 1.0

        self._cape_decision_counter += 1
        self._cape_blocks_since_switch += 1
        if (
            self._cape_decision_counter >= self.config.cape.decision_interval_blocks
            and self._cape_blocks_since_switch >= self._cape_hold_blocks
        ):
            self._cape_decision_counter = 0
            decision = self._cape_policy.select(cues)
            self._cape_decision = decision
            if decision.treatment != self._cape_active:
                self._cape_previous = self._cape_active
                self._cape_active = decision.treatment
                self._cape_transition_position = 0
                self._cape_blocks_since_switch = 0

        alpha = min(1.0, self._cape_transition_position / self._cape_transition_blocks)
        self._cape_transition_position += 1
        if alpha >= 1.0:
            self._cape_previous = self._cape_active
        return self._cape_decision, self._cape_previous, self._cape_active, alpha

    def _strategy_modifier_db(
        self,
        treatment: Treatment,
        base_gain_db: NDArray[np.float64],
        spectrum: NDArray[np.complex128],
        f0_hz: float,
        confidence: float,
    ) -> NDArray[np.float64]:
        modifier = np.zeros_like(base_gain_db)
        music = self.config.music
        strength = treatment.strength
        if treatment.name in {"reference", "transient_preserve"}:
            return modifier
        if treatment.name == "minimal_processing":
            return -0.35 * strength * base_gain_db
        if treatment.name == "harmonic_cue":
            if f0_hz <= 0.0 or confidence < music.confidence_threshold:
                return modifier
            harmonic_number = np.maximum(1.0, np.rint(self._frequencies / f0_hz))
            nearest = harmonic_number * f0_hz
            valid = (harmonic_number <= 20.0) & (self._frequencies >= f0_hz)
            cents = 1_200.0 * np.log2(
                np.maximum(self._frequencies, 1.0) / np.maximum(nearest, 1.0)
            )
            harmonic_weight = np.exp(-0.5 * (cents / 65.0) ** 2) * valid
            confidence_weight = np.clip(
                (confidence - music.confidence_threshold)
                / max(1e-6, 1.0 - music.confidence_threshold),
                0.0,
                1.0,
            )
            return music.harmonic_boost_db * strength * confidence_weight * harmonic_weight
        if treatment.name == "timbre_preserve":
            magnitude = np.abs(spectrum)
            weighted_mean = float(
                np.sum(base_gain_db * magnitude) / max(np.sum(magnitude), 1e-12)
            )
            return np.clip(
                strength * music.timbre_preserve_mix * (weighted_mean - base_gain_db),
                -1.5,
                1.5,
            )
        if treatment.name == "melody_relief":
            magnitude = np.abs(spectrum) + 1e-12
            local = np.convolve(magnitude, np.ones(9, dtype=np.float64) / 9.0, mode="same")
            salience_db = 20.0 * np.log10(magnitude / np.maximum(local, 1e-12))
            salience = np.clip(salience_db / 12.0, 0.0, 1.0)
            band = (self._frequencies >= 150.0) & (self._frequencies <= 5_000.0)
            return music.melody_relief_db * strength * salience * band
        return modifier

    def _gain_curve(
        self,
        band_gain_db: NDArray[np.float64],
        spectrum: NDArray[np.complex128],
        f0_hz: float,
        confidence: float,
        previous: Treatment,
        active: Treatment,
        alpha: float,
    ) -> NDArray[np.float64]:
        gain_db = self._base_gain_curve_db(band_gain_db)
        previous_modifier = self._strategy_modifier_db(
            previous, gain_db, spectrum, f0_hz, confidence
        )
        active_modifier = self._strategy_modifier_db(
            active, gain_db, spectrum, f0_hz, confidence
        )
        gain_db = gain_db + (1.0 - alpha) * previous_modifier + alpha * active_modifier
        music = self.config.music
        gain_db[0] = 0.0
        return np.power(10.0, gain_db / 20.0)

    def _limit(self, output: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        limiter = self.config.limiter
        ceiling = 10.0 ** (limiter.ceiling_dbfs / 20.0)
        peak = float(np.max(np.abs(output))) if output.size else 0.0
        required = min(1.0, ceiling / max(peak, 1e-12))
        if required < self._limiter_gain:
            self._limiter_gain = required
        else:
            release = np.exp(
                -(self.hop / self.sample_rate) / (limiter.release_ms / 1_000.0)
            )
            self._limiter_gain = release * self._limiter_gain + (1.0 - release)
        limited = output * self._limiter_gain
        limited = np.clip(limited, -ceiling, ceiling)
        reduction_db = -_db(max(self._limiter_gain, 1e-12))
        return limited, max(0.0, reduction_db)

    def process_block(self, samples: FloatArray) -> tuple[FloatArray, BlockDiagnostics]:
        started = perf_counter_ns()
        block = np.asarray(samples, dtype=np.float64)
        if block.ndim != 1 or block.size != self.hop:
            raise ValueError(f"expected a mono block with exactly {self.hop} samples")
        fail_safe = bool(np.any(~np.isfinite(block)))
        if fail_safe:
            block = np.zeros_like(block)

        frame = np.concatenate((self._previous_input, block))
        spectrum = rfft(frame * self._window)
        levels = self._band_levels(spectrum)
        f0_hz, f0_confidence = self._pitch(block)
        cues = self._music_cues(spectrum, block, f0_hz, f0_confidence)
        cape, previous_treatment, active_treatment, treatment_alpha = self._select_treatment(cues)
        transient_mix = (
            (1.0 - treatment_alpha)
            * (previous_treatment.strength if previous_treatment.name == "transient_preserve" else 0.0)
            + treatment_alpha
            * (active_treatment.strength if active_treatment.name == "transient_preserve" else 0.0)
        )
        attack_scale = 1.0 + transient_mix * (self.config.music.transient_attack_scale - 1.0)
        gains = self._smooth_gains(self._target_gains(levels), attack_scale=attack_scale)
        modified = spectrum * self._gain_curve(
            gains,
            spectrum,
            f0_hz,
            f0_confidence,
            previous_treatment,
            active_treatment,
            treatment_alpha,
        )
        reconstructed = irfft(modified, n=self.fft_size) * self._window
        output = reconstructed[: self.hop] + self._overlap
        self._overlap = reconstructed[self.hop :].copy()
        self._previous_input = block.copy()

        output, limiter_reduction = self._limit(output)
        ramp_samples = int(round(self.config.limiter.startup_ramp_ms * self.sample_rate / 1_000.0))
        if ramp_samples > 0 and self._processed_samples < ramp_samples:
            positions = self._processed_samples + np.arange(self.hop)
            ramp = np.clip(positions / ramp_samples, 0.0, 1.0)
            output *= ramp
        self._processed_samples += self.hop
        if fail_safe or np.any(~np.isfinite(output)):
            output.fill(0.0)
            fail_safe = True
            self._fail_safe_count += 1

        elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
        diagnostics = BlockDiagnostics(
            runtime_ms=elapsed_ms,
            input_peak_dbfs=_db(float(np.max(np.abs(block))) if block.size else 0.0),
            output_peak_dbfs=_db(float(np.max(np.abs(output))) if output.size else 0.0),
            band_levels_dbfs=levels.astype(np.float32),
            applied_gain_db=gains.astype(np.float32),
            f0_hz=f0_hz,
            f0_confidence=f0_confidence,
            cape_action=cape.treatment.name,
            cape_strength=cape.treatment.strength,
            cape_score=cape.score,
            cape_uncertainty=cape.uncertainty,
            cape_estimated_distortion=cape.estimated_distortion,
            cape_delta_pitch=cape.delta[0],
            cape_delta_timbre=cape.delta[1],
            cape_delta_melody=cape.delta[2],
            cape_delta_naturalness=cape.delta[3],
            cape_delta_clarity=cape.delta[4],
            cape_reason=cape.reason,
            limiter_reduction_db=limiter_reduction,
            fail_safe_triggered=fail_safe,
        )
        return output.astype(np.float32), diagnostics

    def flush(self) -> tuple[FloatArray, BlockDiagnostics]:
        return self.process_block(np.zeros(self.hop, dtype=np.float32))
