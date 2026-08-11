from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hearing_assist.config import load_hearing_profile
from hearing_assist.dsp import HearingAssistProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    config = load_hearing_profile(args.profile)
    processor = HearingAssistProcessor(config)
    rng = np.random.default_rng(20_260_811)
    blocks = int(np.ceil(args.seconds * config.audio.sample_rate / config.audio.block_size))
    runtimes = []
    for _ in range(blocks):
        block = (0.03 * rng.standard_normal(config.audio.block_size)).astype(np.float32)
        _, diagnostics = processor.process_block(block)
        runtimes.append(diagnostics.runtime_ms)
    values = np.asarray(runtimes)
    deadline = processor.algorithmic_latency_ms
    result = {
        "profile": config.name,
        "blocks": blocks,
        "algorithmic_latency_ms": deadline,
        "runtime_ms": {
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        },
        "deadline_misses": int(np.sum(values > deadline)),
        "note": "DSP-only benchmark; excludes audio-driver and hardware latency.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
