from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_ci.config import load_config
from music_ci.core.pipeline import ResearchPipeline
from music_ci.io.audio import generate_demo_melody, read_wav_mono, write_wav_mono
from music_ci.io.results import save_results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-ci",
        description="Research-only cochlear implant music coding simulator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("demo", "process"):
        sub = subparsers.add_parser(command)
        if command == "process":
            sub.add_argument("input", type=Path)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--strategy", choices=("ace", "msa"), default="msa")
        sub.add_argument("--active-channels", type=int, default=8)
        sub.add_argument("--config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    config.selection.strategy = args.strategy
    config.selection.active_channels = args.active_channels
    config.validate()

    if args.command == "demo":
        samples = generate_demo_melody(config.audio.sample_rate)
        write_wav_mono(args.output / "demo_input.wav", samples, config.audio.sample_rate)
    else:
        samples = read_wav_mono(args.input, config.audio.sample_rate)

    pipeline = ResearchPipeline(config)
    result = pipeline.process(samples)
    output = save_results(args.output, result, config, pipeline.filterbank)
    with (output / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    runtime = metrics["runtime_ms"]
    print(f"saved: {output.resolve()}")
    print(
        f"strategy={metrics['strategy']} frames={metrics['frames']} "
        f"mean={runtime['mean']:.3f}ms p99={runtime['p99']:.3f}ms "
        f"deadline_misses={runtime['deadline_misses']}"
    )

