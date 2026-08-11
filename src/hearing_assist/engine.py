from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from music_ci.io.audio import read_wav_mono, write_wav_mono

from .config import HearingAssistConfig
from .dsp import BlockDiagnostics, HearingAssistProcessor


def process_array(
    samples: NDArray[np.float32], config: HearingAssistConfig
) -> tuple[NDArray[np.float32], list[BlockDiagnostics]]:
    processor = HearingAssistProcessor(config)
    hop = config.audio.block_size
    sample_count = int(samples.size)
    padded_count = int(np.ceil(max(1, sample_count) / hop) * hop)
    padded = np.pad(np.asarray(samples, dtype=np.float32), (0, padded_count - sample_count))
    delayed_outputs: list[NDArray[np.float32]] = []
    diagnostics: list[BlockDiagnostics] = []
    for start in range(0, padded_count, hop):
        output, report = processor.process_block(padded[start : start + hop])
        delayed_outputs.append(output)
        diagnostics.append(report)
    tail, report = processor.flush()
    delayed_outputs.append(tail)
    diagnostics.append(report)
    aligned = np.concatenate(delayed_outputs[1:])[:sample_count]
    return aligned.astype(np.float32), diagnostics


def _metrics(
    source: NDArray[np.float32],
    output: NDArray[np.float32],
    diagnostics: list[BlockDiagnostics],
    config: HearingAssistConfig,
    elapsed_sec: float,
) -> dict[str, object]:
    runtimes = np.asarray([item.runtime_ms for item in diagnostics], dtype=np.float64)
    reductions = np.asarray([item.limiter_reduction_db for item in diagnostics], dtype=np.float64)
    actions = [item.cape_action for item in diagnostics]
    action_counts = {name: actions.count(name) for name in sorted(set(actions))}
    deadline = 1_000.0 * config.audio.block_size / config.audio.sample_rate
    return {
        "profile": config.name,
        "duration_sec": source.size / config.audio.sample_rate,
        "wall_time_sec": elapsed_sec,
        "algorithmic_latency_ms": deadline,
        "input_peak_dbfs": float(20 * np.log10(max(float(np.max(np.abs(source))), 1e-12))) if source.size else -240.0,
        "output_peak_dbfs": float(20 * np.log10(max(float(np.max(np.abs(output))), 1e-12))) if output.size else -240.0,
        "limiter_active_block_ratio": float(np.mean(reductions > 0.01)) if reductions.size else 0.0,
        "fail_safe_blocks": int(sum(item.fail_safe_triggered for item in diagnostics)),
        "cape": {
            "enabled": config.cape.enabled,
            "action_block_counts": action_counts,
            "reference_block_ratio": float(np.mean([name == "reference" for name in actions]))
            if actions
            else 1.0,
            "mean_uncertainty": float(np.mean([item.cape_uncertainty for item in diagnostics]))
            if diagnostics
            else 0.0,
            "mean_estimated_distortion": float(
                np.mean([item.cape_estimated_distortion for item in diagnostics])
            )
            if diagnostics
            else 0.0,
        },
        "runtime_ms": {
            "mean": float(np.mean(runtimes)) if runtimes.size else 0.0,
            "p95": float(np.percentile(runtimes, 95)) if runtimes.size else 0.0,
            "p99": float(np.percentile(runtimes, 99)) if runtimes.size else 0.0,
            "max": float(np.max(runtimes)) if runtimes.size else 0.0,
            "block_deadline_ms": deadline,
            "deadline_misses": int(np.sum(runtimes > deadline)),
        },
        "safety_note": "The limiter ceiling is dBFS, not acoustic dB SPL. Hardware calibration and professional verification are still required.",
    }


def process_wav(
    input_path: str | Path,
    output_path: str | Path,
    config: HearingAssistConfig,
    report_dir: str | Path | None = None,
) -> dict[str, object]:
    source = read_wav_mono(input_path, config.audio.sample_rate)
    started = perf_counter()
    output, diagnostics = process_array(source, config)
    elapsed = perf_counter() - started
    write_wav_mono(output_path, output, config.audio.sample_rate)
    report_root = Path(report_dir) if report_dir is not None else Path(output_path).with_suffix("")
    report_root.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(source, output, diagnostics, config, elapsed)
    with (report_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (report_root / "profile_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
    with (report_root / "block_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "block",
                "runtime_ms",
                "input_peak_dbfs",
                "output_peak_dbfs",
                "f0_hz",
                "f0_confidence",
                "cape_action",
                "cape_strength",
                "cape_score",
                "cape_uncertainty",
                "cape_estimated_distortion",
                "cape_delta_pitch",
                "cape_delta_timbre",
                "cape_delta_melody",
                "cape_delta_naturalness",
                "cape_delta_clarity",
                "cape_reason",
                "limiter_reduction_db",
                "fail_safe",
            ]
        )
        for index, item in enumerate(diagnostics):
            writer.writerow(
                [
                    index,
                    item.runtime_ms,
                    item.input_peak_dbfs,
                    item.output_peak_dbfs,
                    item.f0_hz,
                    item.f0_confidence,
                    item.cape_action,
                    item.cape_strength,
                    item.cape_score,
                    item.cape_uncertainty,
                    item.cape_estimated_distortion,
                    item.cape_delta_pitch,
                    item.cape_delta_timbre,
                    item.cape_delta_melody,
                    item.cape_delta_naturalness,
                    item.cape_delta_clarity,
                    item.cape_reason,
                    item.limiter_reduction_db,
                    int(item.fail_safe_triggered),
                ]
            )
    return metrics
