from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from music_ci.config import AppConfig
from music_ci.core.models import PipelineResult
from music_ci.io.audio import write_wav_mono
from music_ci.output.vocoder import synthesize_noise_band_vocoder


def _switch_rate(result: PipelineResult) -> float:
    masks = [frame.selection.selected_mask for frame in result.frame_results]
    if len(masks) < 2:
        return 0.0
    changes = [np.mean(left != right) for left, right in zip(masks[:-1], masks[1:])]
    return float(np.mean(changes))


def save_results(
    output_dir: str | Path,
    result: PipelineResult,
    config: AppConfig,
    filterbank,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "electrodogram.npy", result.electrodogram)

    with (destination / "selected_channels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_sec", "selected_channels_zero_based"])
        for frame in result.frame_results:
            writer.writerow(
                [
                    frame.frame_index,
                    frame.sample_index / result.sample_rate,
                    " ".join(map(str, frame.selection.selected_indices.tolist())),
                ]
            )

    with (destination / "f0_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_sec", "f0_hz", "confidence", "harmonicity"])
        for frame in result.frame_results:
            analysis = frame.analysis
            writer.writerow(
                [
                    frame.frame_index,
                    frame.sample_index / result.sample_rate,
                    analysis.f0_hz,
                    analysis.f0_confidence,
                    analysis.harmonicity,
                ]
            )

    with (destination / "channel_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "channel",
                "selected",
                "final_score",
                "energy_score",
                "harmonic_score",
                "continuity_score",
                "redundancy_penalty",
            ]
        )
        for frame in result.frame_results:
            selection = frame.selection
            for channel in range(config.filterbank.num_channels):
                writer.writerow(
                    [
                        frame.frame_index,
                        channel,
                        int(selection.selected_mask[channel]),
                        selection.scores[channel],
                        selection.energy_scores[channel],
                        selection.harmonic_scores[channel],
                        selection.continuity_scores[channel],
                        selection.redundancy_penalties[channel],
                    ]
                )

    runtimes = np.array([frame.runtime_ms for frame in result.frame_results], dtype=np.float64)
    with (destination / "runtime.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "runtime_ms", "deadline_miss"])
        for frame in result.frame_results:
            writer.writerow([frame.frame_index, frame.runtime_ms, int(frame.deadline_miss)])

    confident_f0 = [
        frame.analysis.f0_hz
        for frame in result.frame_results
        if frame.analysis.f0_confidence >= config.pitch.confidence_threshold
    ]
    metrics = {
        "strategy": config.selection.strategy,
        "frames": len(result.frame_results),
        "duration_sec": result.input_samples.size / result.sample_rate,
        "active_channels": config.selection.active_channels,
        "selection_switch_rate": _switch_rate(result),
        "confident_f0_frame_ratio": len(confident_f0) / max(1, len(result.frame_results)),
        "median_confident_f0_hz": float(np.median(confident_f0)) if confident_f0 else None,
        "runtime_ms": {
            "mean": float(np.mean(runtimes)) if runtimes.size else 0.0,
            "p95": float(np.percentile(runtimes, 95)) if runtimes.size else 0.0,
            "p99": float(np.percentile(runtimes, 99)) if runtimes.size else 0.0,
            "max": float(np.max(runtimes)) if runtimes.size else 0.0,
            "deadline_misses": int(np.sum(runtimes > config.runtime.frame_deadline_ms)),
        },
        "warning": "Research simulation only; not for clinical stimulation.",
    }
    with (destination / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with (destination / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, ensure_ascii=False)

    vocoder = synthesize_noise_band_vocoder(
        result,
        filterbank=filterbank,
        frame_samples=config.audio.frame_samples,
        random_seed=config.runtime.random_seed,
    )
    write_wav_mono(destination / "vocoder.wav", vocoder, result.sample_rate)
    return destination

