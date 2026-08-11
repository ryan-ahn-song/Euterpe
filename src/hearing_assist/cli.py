from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from music_ci.io.audio import generate_demo_melody, write_wav_mono

from .cape import CAPEEffectModel, OUTCOME_NAMES, TREATMENT_NAMES, load_examples
from .config import load_hearing_profile
from .dsp import HearingAssistProcessor
from .engine import process_wav
from .live import LiveHearingAssist, list_audio_devices


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hearing-assist",
        description="Non-clinical real-time music-aware hearing-assist prototype",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("profile-check", help="validate and summarize a profile")
    check.add_argument("--profile", type=Path, required=True)

    offline = commands.add_parser("offline", help="process a PCM WAV file")
    offline.add_argument("input", type=Path)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--profile", type=Path, required=True)
    offline.add_argument("--report-dir", type=Path)

    demo = commands.add_parser("demo", help="process the built-in melody")
    demo.add_argument("--output-dir", type=Path, required=True)
    demo.add_argument("--profile", type=Path, required=True)

    commands.add_parser("devices", help="list PortAudio input/output devices")

    train = commands.add_parser("cape-train", help="fit a CAPE counterfactual outcome ensemble")
    train.add_argument("--dataset", type=Path, required=True, help="CAPE JSONL observations")
    train.add_argument("--output", type=Path, required=True, help="output .npz model")
    train.add_argument("--ensemble-size", type=int, default=7)
    train.add_argument("--hidden-features", type=int, default=96)
    train.add_argument("--ridge", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=20_260_811)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument(
        "--data-status",
        choices=("unknown", "synthetic", "human-research", "clinical"),
        default="unknown",
        help="required provenance label; this is stored in the model",
    )

    evaluate = commands.add_parser("cape-evaluate", help="evaluate a fitted CAPE model")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)

    inspect = commands.add_parser("cape-inspect", help="show CAPE model schema and training metadata")
    inspect.add_argument("--model", type=Path, required=True)

    live = commands.add_parser("live", help="run microphone-to-wired-output streaming")
    live.add_argument("--profile", type=Path, required=True)
    live.add_argument("--input-device")
    live.add_argument("--output-device")
    live.add_argument("--seconds", type=float)
    live.add_argument(
        "--accept-uncalibrated-risk",
        action="store_true",
        help="required unless the profile records a calibrated output path",
    )
    return parser


def _print_profile(config) -> None:
    processor = HearingAssistProcessor(config)
    print(f"profile: {config.name}")
    print(f"description: {config.description}")
    print(f"sample rate/block: {config.audio.sample_rate} Hz/{config.audio.block_size}")
    print(f"algorithmic latency: {processor.algorithmic_latency_ms:.2f} ms")
    print(f"band gain dB: {config.prescribed_gain_db().round(2).tolist()}")
    print(f"digital ceiling: {config.limiter.ceiling_dbfs:.1f} dBFS")
    print(f"hardware calibrated: {config.calibration.output_calibrated}")
    print(f"CAPE enabled: {config.cape.enabled}")
    if config.cape.enabled:
        print(f"CAPE model: {config.cape.model_path}")
        print(f"CAPE listener: {config.cape.listener_profile_path}")


def _split_examples(examples, validation_fraction: float, seed: int):
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in [0, 0.5)")
    if validation_fraction == 0.0 or len(examples) < 20:
        return list(examples), [], "none"
    rng = np.random.default_rng(seed)
    listener_ids = sorted({item.listener.listener_id for item in examples})
    if len(listener_ids) >= 3:
        count = max(1, min(len(listener_ids) - 1, int(np.ceil(validation_fraction * len(listener_ids)))))
        held_out = set(rng.choice(listener_ids, size=count, replace=False).tolist())
        train = [item for item in examples if item.listener.listener_id not in held_out]
        validation = [item for item in examples if item.listener.listener_id in held_out]
        return train, validation, "listener-held-out"
    indices = rng.permutation(len(examples))
    count = max(1, int(np.ceil(validation_fraction * len(examples))))
    validation_indices = set(int(index) for index in indices[:count])
    train = [item for index, item in enumerate(examples) if index not in validation_indices]
    validation = [item for index, item in enumerate(examples) if index in validation_indices]
    return train, validation, "observation-held-out"


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "devices":
        try:
            print(list_audio_devices())
        except RuntimeError as exc:
            parser.error(str(exc))
        return

    if args.command == "cape-train":
        try:
            examples = load_examples(args.dataset)
            train_examples, validation_examples, split = _split_examples(
                examples, args.validation_fraction, args.seed
            )
            model = CAPEEffectModel(
                ensemble_size=args.ensemble_size,
                hidden_features=args.hidden_features,
                ridge=args.ridge,
                seed=args.seed,
            )
            training_metrics = model.fit(train_examples)
            validation_metrics = model.evaluate(validation_examples) if validation_examples else None
            model.training_metadata.update(
                {
                    "dataset_file": args.dataset.name,
                    "split": split,
                    "training_observations": len(train_examples),
                    "validation_observations": len(validation_examples),
                    "validation_metrics": validation_metrics,
                    "data_status": args.data_status,
                }
            )
            model.save(args.output)
        except (OSError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            parser.error(str(exc))
        print(f"saved: {args.output.resolve()}")
        print(
            json.dumps(
                {
                    "split": split,
                    "training": training_metrics,
                    "validation": validation_metrics,
                    "warning": "Prediction accuracy is not evidence of hearing benefit without held-out listener testing.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "cape-evaluate":
        try:
            model = CAPEEffectModel.load(args.model)
            metrics = model.evaluate(load_examples(args.dataset))
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.command == "cape-inspect":
        try:
            model = CAPEEffectModel.load(args.model)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "ensemble_size": model.ensemble_size,
                    "hidden_features": model.hidden_features,
                    "treatments": TREATMENT_NAMES,
                    "outcomes": OUTCOME_NAMES,
                    "training": model.training_metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    config = load_hearing_profile(args.profile)
    if args.command == "profile-check":
        _print_profile(config)
        return

    if args.command == "offline":
        metrics = process_wav(
            args.input,
            args.output,
            config,
            report_dir=args.report_dir,
        )
        print(f"saved: {args.output.resolve()}")
        print(json.dumps(metrics["runtime_ms"], ensure_ascii=False))
        return

    if args.command == "demo":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        source_path = args.output_dir / "demo_input.wav"
        output_path = args.output_dir / "demo_hearing_assist.wav"
        # Keep the built-in source well below the limiter so the example reveals
        # frequency-dependent gain instead of mostly demonstrating peak control.
        samples = (0.2 * generate_demo_melody(config.audio.sample_rate)).astype("float32")
        write_wav_mono(source_path, samples, config.audio.sample_rate)
        metrics = process_wav(
            source_path,
            output_path,
            config,
            report_dir=args.output_dir / "report",
        )
        print(f"input: {source_path.resolve()}")
        print(f"output: {output_path.resolve()}")
        print(json.dumps(metrics["runtime_ms"], ensure_ascii=False))
        return

    if args.command == "live":
        if not config.calibration.output_calibrated and not args.accept_uncalibrated_risk:
            parser.error(
                "this profile is not acoustically calibrated; first lower the physical output "
                "volume, then pass --accept-uncalibrated-risk for bench testing"
            )
        _print_profile(config)
        session = LiveHearingAssist(
            config,
            input_device=args.input_device,
            output_device=args.output_device,
        )
        try:
            summary = session.run(duration_sec=args.seconds)
        except RuntimeError as exc:
            parser.error(str(exc))
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
